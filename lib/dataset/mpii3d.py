
from lib.dataset import Dataset3D
from lib.core.config import MPII3D_DIR


class MPII3D(Dataset3D):
    def __init__(self, set, seqlen, overlap=0, debug=False):
        db_name = 'mpii3d'

        # during testing we don't need data augmentation
        # but we can use it as an ensemble
        is_train = set == 'train'
        overlap = overlap if is_train else 0.
        print('MPII3D Dataset overlap ratio: ', overlap)
        super(MPII3D, self).__init__(
            set = set,
            folder=MPII3D_DIR,
            seqlen=seqlen,
            overlap=overlap,
            dataset_name=db_name,
            debug=debug,
        )
        print(f'{db_name} - number of dataset objects {self.__len__()}')