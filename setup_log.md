During monst3r requirements installation:

# Note that many libraries overlap with mast3r conda environment so I decided to build the monst3r env on top of the mast3r environment to avoid storage space redundancy
# but is also risky in that this changes some versions (like numpy and torch versions that is used in mast3r). we shall see
# below is the error message when ```pip install -r requirements.txt``` from monst3r repo.

    Attempting uninstall: torch
        Found existing installation: torch 2.0.1
        Uninstalling torch-2.0.1:
        Successfully uninstalled torch-2.0.1
    Attempting uninstall: torchvision
        Found existing installation: torchvision 0.15.2
        Uninstalling torchvision-0.15.2:
        Successfully uninstalled torchvision-0.15.2
    ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    torchaudio 2.0.2 requires torch==2.0.1, but you have torch 2.5.0 which is incompatible.