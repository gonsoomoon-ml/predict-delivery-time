

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    from prettytable import PrettyTable
    from autogluon import TabularPrediction as task


# ------------------------------------------------------------ #
# Hosting methods                                              #
# ------------------------------------------------------------ #

def model_fn(model_dir):
    """
    Load the gluon model. Called once when hosting service starts.
    :param: model_dir The directory where model files are stored.
    :return: a model (in this case an AutoGluon network)
    """
    net = task.load(model_dir)
    return net


def transform_fn(models, data, input_content_type, output_content_type):
    """
    Transform a request using the Gluon model. Called once per request.
    :param models: The Gluon model and the column info.
    :param data: The request payload.
    :param input_content_type: The request content type. ('text/csv')
    :param output_content_type: The (desired) response content type. ('text/csv')
    :return: response payload and content type.
    """
    print("=========== Start in transform_fn ================")
#    start = timer()
    net = models[0]
    column_dict = models[1]
    
    print("infer2: Gonsoo Test 01 in transform_fn()")

    # text/csv
    if input_content_type == 'text/csv':
        
        # Load dataset
        columns = column_dict['columns']
        df = pd.read_csv(StringIO(data), header=None)
        df_preprosessed = preprocess(df, columns, net.label_column)
        ds = task.Dataset(df=df_preprosessed)
        
        try:
            predictions = net.predict(ds)
        except:
            try:
                predictions = net.predict(ds.fillna(0.0))
                warnings.warn('Filled NaN\'s with 0.0 in order to predict.')
            except Exception as e:
                response_body = e
                return response_body, output_content_type
        
        # Print prediction counts, limit in case of regression problem
        pred_counts = Counter(predictions.tolist())
        n_display_items = 30
        if len(pred_counts) > n_display_items:
            print(f'Top {n_display_items} prediction counts: '
                  f'{dict(take(n_display_items, pred_counts.items()))}')
        else:
            print(f'Prediction counts: {pred_counts}')

        # Form response
        output = StringIO()
        pd.DataFrame(predictions).to_csv(output, header=False, index=False)
        response_body = output.getvalue() 

        # If target column passed, evaluate predictions performance
        target = net.label_column
        if target in ds:
            print(f'Label column ({target}) found in input data. '
                  'Therefore, evaluating prediction performance...')    
            try:
                performance = net.evaluate_predictions(y_true=ds[target], 
                                                       y_pred=predictions, 
                                                       auxiliary_metrics=True)                
                print(json.dumps(performance, indent=4, default=pd.DataFrame.to_json))
                time.sleep(0.1)
            except Exception as e:
                # Print exceptions on evaluate, continue to return predictions
                print(f'Exception: {e}')
    else:
        raise NotImplementedError("content_type must be 'text/csv'")

    elapsed_time = round(timer()-start,3)
    print(f'Elapsed time: {round(timer()-start,3)} seconds')           
    
    return response_body, output_content_type



