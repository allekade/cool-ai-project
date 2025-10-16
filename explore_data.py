import pandas as pd

from ucimlrepo import fetch_ucirepo
from ydata_profiling import ProfileReport

def main():
    dataset = fetch_ucirepo(id=327) 
    X = dataset.data.features 
    y = dataset.data.targets

    df = pd.concat([X, y], axis=1)
    profile = ProfileReport(df, title="Profiling Report")
    

    profile.to_file("websiteinfo.html")

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)



if __name__ == "__main__":
    main()