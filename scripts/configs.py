DEBUG_CONFIG = {
    'DEVICE': 'cpu',
    'ROW_GLAM_TYPE': 'main',
    'NAME_DATASET': 'publaynet',
    'NAME_TEST_DATASET': 'publaynet',
    'DATASET_PATH': 'debug_data/pdfs',
    'COCO_PATH': 'debug_data/publaynet_mini.json',
    'TEST_PATH': 'debug_data/pdfs',
    'TEST_COCO_PATH': 'debug_data/publaynet_mini.json',
    'CASH_PDF_PATH': 'debug_data/cache',
    'EPOCHS': 2,
}

LOCAL_CONFIG = {
    'DEVICE': 'cpu',
    'ROW_GLAM_TYPE': 'main',
    'NAME_DATASET': 'publaynet',
    'NAME_TEST_DATASET': 'doclaynet',
    'DATASET_PATH': '/Users/macbookair/Downloads/micro_publaynet/pdfs/train',
    'COCO_PATH': '/Users/macbookair/Downloads/micro_publaynet/publaynet/train.json',
    'TEST_PATH': '/Users/macbookair/Downloads/micro_publaynet/pdfs/dev',
    'TEST_COCO_PATH': '/Users/macbookair/Downloads/micro_publaynet/publaynet/val.json',
    'CASH_PDF_PATH': '/Users/macbookair/Downloads/micro_publaynet/tmp/cache_miner',
    'EPOCHS': 30,
}

SERVER_CONFIG = {
    'DEVICE': 'cuda',
    'ROW_GLAM_TYPE': 'main',
    'NAME_DATASET': 'publaynet',
    'NAME_TEST_DATASET': 'publaynet',
    'DATASET_PATH': '/home/daniil/disk01_1TB/datasets/publaynet_pdfs/pdfs/train',
    'COCO_PATH': '/home/daniil/disk01_1TB/datasets/publaynet/train.json',
    'TEST_PATH': '/home/daniil/disk01_1TB/datasets/publaynet_pdfs/pdfs/dev',
    'TEST_COCO_PATH': '/home/daniil/disk01_1TB/datasets/publaynet/val.json',
    'CASH_PDF_PATH': '/home/daniil/disk01_1TB/datasets/tmp/cache_miner_publaynet',
    'EPOCHS': 30,
}

ALL_CONFIGS = {
    'debug': DEBUG_CONFIG,
    'local': LOCAL_CONFIG,
    'server': SERVER_CONFIG,
}
