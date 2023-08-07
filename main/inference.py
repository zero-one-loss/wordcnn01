import pickle
import sys
sys.path.append('..')
import readline

def map_(label):

    return 'positive' if label == 1 else 'negative'


def prediction(sentence, model):

    ts = sentence.split()
    print(f'This review is {map_(model.text_pred(ts).argmax(dim=1).item())}.')


with open('checkpoints/mr_wordcnn01_bp_01_fs_2.pkl', 'rb') as f:
    model_01 = pickle.load(f)

with open('checkpoints/mr_wordcnn01_bp_relu_2.pkl', 'rb') as f:
    model_bp = pickle.load(f)


def evaluation(text):

    print('review: ', text)
    print('model_bp: ')
    prediction(text, model_bp)
    print('model_01: ')
    prediction(text, model_01)
    print()


if __name__ == '__main__':

    # Case 1
    # text = 'has the feel of an unedited personal journal .'
    #
    # evaluation(text)
    #
    # text = 'has the feel of an uncensored personal journal .'
    #
    # evaluation(text)

    # Case 2
    # text = "a terrific date movie , whatever your orientation"
    #
    # evaluation(text)
    #
    # text = "a exceptional date movie , whatever your orientation"
    #
    # evaluation(text)

    while True:
        print('Type your review, press Enter when typing is done')
        text = input()
        print()
        evaluation(text)
