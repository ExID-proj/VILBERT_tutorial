from .retreival_dataset import RetreivalDataset, RetreivalDatasetVal, RetreivalDatasetTrans


__all__ = [
    "RetreivalDataset",
    "RetreivalDatasetVal",
]

DatasetMapTrain = {
    "RetrievalFlickr30k": RetreivalDataset,
}


DatasetMapEval = {
    "RetrievalFlickr30k": RetreivalDatasetVal,
}


DatasetMapTrans = {
    "RetrievalFlickr30k": RetreivalDatasetTrans,
}
