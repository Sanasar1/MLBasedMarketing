from joblib import load

def load_model():
    model = load('/Users/sanasar/PycharmProjects/MLBasedMarketing/MLBasedMarketing/fileupload/model/kmeans.pkl')
    return model