! {{EXCITED_METHOD}}
! {{SOLVENT}}
%basis
  NewGTO Sb "SARC-ZORA-TZVP" end
  NewGTO I  "SARC-ZORA-TZVP" end
  NewGTO Te "SARC-ZORA-TZVP" end
end
%tddft
  NRoots {{NROOTS}}
  Triplets true
  dosoc true
end
%pal nprocs {{NPROCS}} end
%maxcore {{MAXCORE_MB}}
%base "{{FILE_NUMBER}}_td"
* xyzfile {{CHARGE}} {{MULTIPLICITY}} {{FILE_NUMBER}}.xyz

