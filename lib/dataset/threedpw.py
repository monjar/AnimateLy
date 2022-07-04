
from lib.dataset import Dataset3D
from lib.core.config import THREEDPW_DIR

class ThreeDPW(Dataset3D):
    def __init__(self, set, seqlen, overlap=0.75, debug=False):
        db_name = '3dpw'

        # during testing we don't need data augmentation
        # but we can use it as an ensemble
        is_train = False
        overlap = overlap if is_train else 0.
        print('3DPW Dataset overlap ratio: ', overlap)
        super(ThreeDPW, self).__init__(
            set=set,
            folder=THREEDPW_DIR,
            seqlen=seqlen,
            overlap=overlap,
            dataset_name=db_name,
            debug=debug,
        )
        print(f'{db_name} - number of dataset objects {self.__len__()}')