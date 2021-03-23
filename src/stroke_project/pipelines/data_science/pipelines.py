from kedro.pipeline import Pipeline, node
from .nodes import split_data, over_sample_data, train_model, evaluate_model

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=split_data,
                inputs=['preprocessed_stroke_data', 'parameters'],
                outputs=['X_train', 'X_test', 'y_train', 'y_test'],
                name='split_data_node'
            ),
            node(
                func=over_sample_data,
                inputs=['X_train', 'y_train', 'parameters'],
                outputs=['X_train_res', 'y_train_res'],
                name='over_sample_node'
            ),
            node(
                func=train_model,
                inputs=['X_train_res', 'y_train_res'],
                outputs='lgbm',
                name='train_model_node'
            ),
            node(
                func=evaluate_model,
                inputs=['lgbm', 'X_test', 'y_test'],
                outputs= None,
                name='evaluate_model_node'
            )
        ]
    )