"""Paper prompts on the main package LLM backend; no persistent API cache."""

import json
import re
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from rdkit import Chem
from .settings import P
from .references import SuccessfulExamplesManager
from .prompts import EST_DEFINITION, inverse_design_prompt_optimized, comprehensive_scoring_prompt


class APIError(RuntimeError):
    pass


class Client:
    def __init__(self, backend, model="configured-backend"):
        self.backend = backend
        self.model = model

    def call(self, prompt, temperature, scope, max_tokens=12000):
        # Transport errors propagate; never substitute a different algorithm.
        return self.backend.complete(prompt, temperature, max_tokens)

    def targets(self, profiles, scope, no_context=False):
        def generate(item):
            i, profile = item
            if no_context:
                prompt = (
                    EST_DEFINITION
                    + "Propose one valid molecular SMILES for these numerical targets: S1 between 1.3 and 3.2 eV, EST < 0.3 eV; maximize f and SOCME/EST. Focus: "
                    + profile["name"]
                    + ". Return only one SMILES."
                )
            else:
                prompt = inverse_design_prompt_optimized(profile, [])
            invalid = []
            for attempt in range(10):
                content = self.call(
                    prompt + ("\nAvoid invalid proposals: " + str(invalid) if invalid else ""),
                    0.2,
                    scope + f"/target{i}/attempt{attempt}",
                    4000,
                )
                for line in content.strip().splitlines():
                    candidate = line.strip().strip(chr(96)).strip('"').strip("'")
                    candidate = re.sub(r"^(SMILES\s*[:=]\s*)", "", candidate, flags=re.I)
                    mol = Chem.MolFromSmiles(candidate)
                    if mol is not None:
                        return Chem.MolToSmiles(mol)
                invalid.append(content[:300])
            raise APIError("No valid generated target: " + scope)

        with ThreadPoolExecutor(max_workers=3) as pool:
            return list(pool.map(generate, enumerate(profiles)))

    def evaluate(self, lib, history, ids, scope, no_context=False):
        manager = SuccessfulExamplesManager()
        prompts = []
        for id in ids:
            smiles = lib.smiles[int(id)]
            refs = (
                manager.get_scoring_references(
                    history, smiles, global_k=3, local_positive_k=2, local_negative_k=4
                )
                if not no_context
                else {"global_frontier": [], "local_positive": [], "local_negative": []}
            )
            prompt = comprehensive_scoring_prompt(
                smiles, refs["global_frontier"], refs["local_positive"], refs["local_negative"]
            )
            prompts.append({"id": int(id), "prompt": prompt})
        chunks = [
            prompts[i : i + P["api_evaluation_batch"]]
            for i in range(0, len(prompts), P["api_evaluation_batch"])
        ]

        def evaluation(task):
            trial, chunk_idx, chunk = task
            prompt = (
                'Evaluate each target independently using only its own supplied references. Override per-target output formatting: return one JSON array of objects {"id":integer,"values":[S1,EST,f,SOCME/EST,score]}. Include every requested id exactly once, no explanation.\n'
                + json.dumps(chunk, ensure_ascii=False)
            )
            last = None
            lookup = {}
            wanted = [row["id"] for row in chunk]
            for repair in range(8):
                pending = [row for row in chunk if row["id"] not in lookup]
                if repair:
                    prompt = (
                        'Previous batch omitted these IDs. Evaluate ONLY the following remaining targets. Return JSON array of {"id":integer,"values":[S1,EST,f,SOCME/EST,score]}; include every listed ID exactly once.\n'
                        + json.dumps(pending, ensure_ascii=False)
                    )
                content = self.call(
                    prompt,
                    P["api_temperatures"][trial],
                    scope + f"/trial{trial}/chunk{chunk_idx}/repair{repair}",
                )
                try:
                    text = content.strip().strip(chr(96)).removeprefix("json").strip()
                    obj = json.loads(text[text.find("[") : text.rfind("]") + 1])
                    returned = [int(row["id"]) for row in obj]
                    if len(returned) != len(set(returned)) or not set(returned).issubset(
                        row["id"] for row in pending
                    ):
                        raise ValueError("Duplicate/unrequested id")
                    for row in obj:
                        value = np.asarray(row["values"], dtype=float)
                        if (
                            value.shape != (5,)
                            or not np.isfinite(value).all()
                            or value[4] < 0
                            or value[4] > 10
                        ):
                            raise ValueError("Invalid prediction shape/nonfinite/score")
                        lookup[int(row["id"])] = value
                    if set(lookup) != set(wanted):
                        raise ValueError("Missing id; request remaining IDs")
                    values = np.stack([lookup[id] for id in wanted])
                    if values.shape != (len(wanted), 5) or not np.isfinite(values).all():
                        raise ValueError("Invalid prediction shape/nonfinite")
                    # Preserve nonphysical property predictions as evaluator errors; do not resample to hide them.
                    return trial, chunk_idx, values[:, :4]
                except Exception as error:
                    last = type(error).__name__ + ": " + str(error)
            raise APIError("Prediction parse failed " + scope + ": " + str(last))

        tasks = [(trial, j, chunk) for trial in range(3) for j, chunk in enumerate(chunks)]
        with ThreadPoolExecutor(max_workers=min(len(tasks), 12)) as pool:
            results = list(pool.map(evaluation, tasks))
        draws = np.stack(
            [
                np.concatenate([v for t, c, v in sorted(results, key=lambda r: r[1]) if t == trial])
                for trial in range(3)
            ]
        )
        return draws, {
            "kind": "llm",
            "trials": 3,
            "temperature_schedule": P["api_temperatures"],
            "prompt_batch": P["api_evaluation_batch"],
            "uncertainty": "response variability (ddof=0)",
        }
