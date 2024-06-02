from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import shap
import uncertainty_based_multimodal

def train_classifier:
    X_train, X_test, y_train, y_test = train_test_split(referral_ornot, referral_label, 
                                                    stratify=referral_label, test_size=0.2, random_state=42)
    RF = RandomForestClassifier(n_estimators=5, random_state=0, max_depth=10, min_samples_leaf=4)

    RF.fit(X_train, y_train)
    return RF, X_test
def explain_model(X_test):
    explainer = shap.TreeExplainer(RF)
    shap_values = explainer.shap_values(np.array(X_test))
    fig=plt.gcf()
    shap.summary_plot(shap_values[0], np.array(X_test), max_display=10, 
                      class_inds="original", class_names=RF.classes_, 
                      feature_names=helpers.get_region_names())

def main():
    
    model, x_test = train_classifier(uncertainty_based_multimodal.referral_ornot, uncertainty_based_multimodal.referral_label)
    explain_model(model, x_test)

if __name__ == "__main__":
    main()