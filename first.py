from ucimlrepo import fetch_ucirepo
import pandas as pd

def load_phishing_data():  
    # fetch dataset 
    phishing_websites = fetch_ucirepo(id=327) 
  
    # data (as pandas dataframes) 
    X = phishing_websites.data.features 
    y = phishing_websites.data.targets 
  
    return X, y

if __name__ == "__main__":
    X, y = load_phushing_data()
    print("shape of X:", X.shape)
    print("Target shape:", y.shape)
