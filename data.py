import datasets

def process_data_and_args(args):
    # load dset
    if args.dataset == 'tweet_eval':
        dataset = datasets.load_dataset('tweet_eval', 'hate')
    else:
        dataset = datasets.load_dataset(args.dataset)
        
    # process dset
    if args.dataset == 'sst2':
        del dataset['test'] # speed things up for now
        args.dataset_key_text = 'sentence'
    elif args.dataset == 'imdb':
        del dataset['unsupervised'] # speed things up for now
        dataset['validation'] = dataset['test']
        del dataset['test']
        args.dataset_key_text = 'text'
    elif args.dataset == 'emotion':
        del dataset['test'] # speed things up for now
        args.dataset_key_text = 'text'
    elif args.dataset == 'rotten_tomatoes':
        del dataset['test'] # speed things up for now
        args.dataset_key_text = 'text'       
    elif args.dataset == 'tweet_eval':
        del dataset['test'] # speed things up for now
        args.dataset_key_text = 'text'               
    #if args.subsample > 0:
    #    dataset['train'] = dataset['train'].select(range(args.subsample))
    return dataset, args