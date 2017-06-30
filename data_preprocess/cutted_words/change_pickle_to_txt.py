import pickle


def change_pickle_to_txt(pickle_file):
    file = open(pickle_file+'.txt', 'w')
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
        # for line in f.readlines():
        #     content = pickle.load(line)
        #     print(content)
        # file.write(content)

    print('done!')

change_pickle_to_txt('train_content.pickle')

