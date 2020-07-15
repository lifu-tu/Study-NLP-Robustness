from .dataset import SNLIDataset, MNLIDataset, QNLIDataset, \
    SNLIHaohanDataset, MNLIStressTestDataset, MNLIHansDataset, MNLIHansTrainDataset, \
    SNLIBreakDataset, SNLISwapDataset, MNLISwapDataset, PAWSDataset, PAWSallDataset, PAWSWikiDataset, PAWSSilverDataset ,QQPlifuDataset, QQPDataset, SNLILifuDataset

tasks = {
    'MNLI': MNLIDataset,
    'QQP-lifu':QQPlifuDataset,
    'QQP':QQPDataset,
    'PAWS': PAWSDataset,
    'PAWSall': PAWSallDataset,
    'PAWS-wiki': PAWSWikiDataset,
    'PAWSSilver':PAWSSilverDataset,
    'MNLI-stress': MNLIStressTestDataset,
    'MNLI-hans': MNLIHansDataset,
    'MNLI-hans-T': MNLIHansTrainDataset,
    'SNLI': SNLIDataset,
    'SNLI-lifu': SNLILifuDataset,
    'SNLI-haohan': SNLIHaohanDataset,
    'SNLI-break': SNLIBreakDataset,
    'SNLI-swap': SNLISwapDataset,
    'MNLI-swap': MNLISwapDataset,
    'QNLI':QNLIDataset,
}

