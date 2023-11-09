podman build . -t deepmol_case_study
podman run --rm -v /home/jcapela/deepmol_case_studies/scripts/:/workspace/scripts/:z -d --device nvidia.com/gpu=0 --security-opt=label=disable --name deepmol_case_study -t deepmol_case_study
#podman run --rm -v /home/jcapela/deepmol_case_studies/scripts/:/workspace/scripts/:z -d --name deepmol_case_study_sklearn -t deepmol_case_study
#podman run --rm -v /home/jcapela/deepmol_case_studies/scripts/:/workspace/scripts/:z -d --device nvidia.com/gpu=1 --security-opt=label=disable --name deepmol_case_study_keras_for_real -t deepmol_case_study
#podman run --rm -v /home/jcapela/deepmol_case_studies/scripts/:/workspace/scripts/:z -d --device nvidia.com/gpu=2 --security-opt=label=disable --name deepmol_case_study_deepchem -t deepmol_case_study
