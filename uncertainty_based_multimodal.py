import helpers

def cos_sim(a, b):
    cos_sim = dot(a, b)/(norm(a)*norm(b))
    return cos_sim
def get_uncertainty(model_unimodal, model_multimodal):
    
    _, uncertainty_m, _, _, _, _ = helpers.uncertainty(model_multimodal, mri_volume, False, SPLIT, x_train_left, x_train_right)

    _, uncertainty_u, _, _, _, _ = helpers.uncertainty(model_unimodal, mri_volume, True, SPLIT, x_train_left, x_train_right)
    
    return uncertainty_u, uncertainty_m

def unimodal_to_multimodal_switch(uncertainty_u, model_u, model_m):
    # switches between unimodal and multimodal models based on uncertainty of unimodal
    
    # get embeddings to compute distance (cos_sim())
    penultimate_layer = tf.keras.models.Model(inputs=model_u.input, outputs=model_u.get_layer(model_u.layers[-3].name).output)

    merged_models_accuracies = []
    imaging_cost = []
    referral_ornot = []
    referral_label = []

    for th in np.arange(0.1, 0.95, 0.05):
        y_pred = []
        uni_high_unc_misclassify = []
        uni_low_unc_classify = []
        for i in range(len(uncertainty_u)):
            output = model_u(np.expand_dims(mri_volume_train[i], axis=0))
            y_pred.append(np.argmax(output[0]))
            if(uncertainty_u[i] > th):
                uni_high_unc_misclassify.append(i)
            if(uncertainty_u[i] <= th and y_true[i] == y_pred[i]):
                uni_low_unc_classify.append(i)
        low_unc_classify = np.array(mri_volume_test)[uni_low_unc_classify]

        embedding_uni_high_unc_misclassify = np.array(penultimate_layer(np.array(mri_volume_test)[uni_high_unc_misclassify]))
        embedding_uni_high_unc_misclassify_average = np.average(embedding_uni_high_unc_misclassify, axis=0)

        embedding_uni_uni_low_unc_classify = np.array(penultimate_layer(np.array(mri_volume_test)[uni_low_unc_classify]))
        embedding_uni_uni_low_unc_classify_average = np.average(embedding_uni_uni_low_unc_classify, axis=0)

        correct_preds = []
        imaging_ctr = 0
        for i, test in enumerate(mri_volume_test):
            embedding_test = np.squeeze(np.array(penultimate_layer(np.expand_dims(test, axis=0))))
            cs_low_uncert = cos_sim(embedding_uni_uni_low_unc_classify_average, embedding_test) 
            cs_high_uncert = cos_sim(embedding_uni_high_unc_misclassify_average, embedding_test)


            if(cs_low_uncert > cs_high_uncert):
                # use uni modal
                output = model_u(np.expand_dims(test, axis=0))
                y_pred= np.argmax(output[0])
                correct_preds.append(y_pred==y_true[i])
                imaging_ctr += 1
                referral_ornot.append(test)
                referral_label.append('Non-referral')
            else:
                # use multi modal
                output = model_m([np.expand_dims(x_left_test[i], axis=0), np.expand_dims(x_right_test[i], axis=0), 
                                      np.expand_dims(test, axis=0)])
                y_pred= np.argmax(output[0])
                correct_preds.append(y_pred==y_true[i])
                imaging_ctr += 2
                referral_ornot.append(test)
                referral_label.append('Non-referral')
        acc = correct_preds.count(True)/len(x_right_test)
        merged_models_accuracies.append(acc)
        imaging_cost.append(imaging_ctr/len(x_right_test))
    return merged_models_accuracies, imaging_cost

def main():
    
    model_unimodal = tf.keras.models.load_model("./models/unimodal.keras", compile=False)
    model_multimodal = tf.keras.models.load_model("./models/multimodal.keras", compile=False)
    
    uncertainty_unimodal, uncertainty_multimodal = get_uncertainty(model_unimodal, model_multimodal)
    
    merged_models_accuracies, imaging_cost = unimodal_to_multimodal_switch(uncertainty_unimodal, model_unimodal, model_multimodal)
if __name__ == "__main__":
    main()
