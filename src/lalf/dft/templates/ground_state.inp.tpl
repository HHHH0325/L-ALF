! {{GROUND_METHOD}}
! {{SOLVENT}}
%pal nprocs {{NPROCS}} end
%maxcore {{MAXCORE_MB}}
%base "{{FILE_NUMBER}}"
%output
  printlevel normal
end
%basis
  PCDTrimAuxJ Coulomb
  PCDThresh -1
end
* xyzfile {{CHARGE}} {{MULTIPLICITY}} a{{FILE_NUMBER}}.xyz

