import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import logging
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

class CKDModelPipeline:
    """
    A class-based pipeline for CKD classification model development.
    This pipeline includes data preprocessing, model training, and evaluation.
    """
    def __init__(self, output_dir="output"):
        """
        Initialize the pipeline with the output directory.
        """
        script_dir = os.path.dirname(__file__)
        self.data_path = os.path.join(script_dir, "kidney_disease.csv")
        self.output_dir = os.path.join(script_dir, output_dir)  # Save output relative to the script
        self.df = None
        os.makedirs(self.output_dir, exist_ok=True)  # Create output directory if it doesn't exist
        self.logger = self.setup_logger(self.output_dir)  # Pass output_dir to logger setup
        self.models = {}
        self.best_models = {}
        self.logger.info(f"Pipeline initialized with data_path='{self.data_path}' and output_dir='{self.output_dir}'")



    @staticmethod
    def setup_logger(output_dir):
        """
        Set up the logger to record events and errors into a log file.
        """
        log_path = os.path.join(output_dir, "pipeline_log.txt")  # Save in output directory
        logger = logging.getLogger('CKDModelPipeline')
        logger.setLevel(logging.INFO)

        # Avoid duplicate handlers
        if not logger.handlers:
            handler = logging.FileHandler(log_path, mode='w')  # Overwrite log file on each run
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        print(f"Logger initialized and writing to {log_path}")  # Debug confirmation
        return logger



    def load_data(self):
        """
        Load the dataset and standardize column names for consistency.
        """
        try:
            # Verify the file exists before attempting to load
            if not os.path.exists(self.data_path):
                raise FileNotFoundError(f"The file {self.data_path} does not exist. Please check the path.")
            self.logger.info("Loading dataset...")
            self.df = pd.read_csv(self.data_path)
            self.df.columns = self.df.columns.str.strip().str.lower().str.replace(' ', '_')
            self.logger.info(f"Dataset loaded successfully with {len(self.df)} rows and {len(self.df.columns)} columns.")
        except Exception as e:
            self.logger.error(f"Error in loading data: {e}")
            raise

    def preprocess_data(self):
        """
        Handle missing data and convert categorical columns into numeric format.
        """
        try:
            self.logger.info("Starting data preprocessing...")
            # Fill missing values for each column
            for col in self.df.columns:
                if self.df[col].dtype == 'object':  # If the column is categorical (text-based)
                    # Replace missing values with the most frequent value (mode)
                    self.df[col].fillna(self.df[col].mode()[0], inplace=True)
                    self.logger.info(f"Filled missing values in column '{col}' with the mode: {self.df[col].mode()[0]}")
                else:  # If the column is numeric
                    # Replace missing values with the average (mean)
                    self.df[col].fillna(self.df[col].mean(), inplace=True)
                    self.logger.info(f"Filled missing values in column '{col}' with the mean: {self.df[col].mean()}")

            # Convert text-based (categorical) columns into numbers using LabelEncoder
            label_enc = LabelEncoder()
            for col in self.df.select_dtypes(include=['object']).columns:
                self.df[col] = label_enc.fit_transform(self.df[col])
                self.logger.info(f"Encoded column '{col}' with LabelEncoder to convert categories into numbers.")

            self.logger.info("Data preprocessing complete.")
        except Exception as e:
            self.logger.error(f"Error in data preprocessing: {e}")
            raise

    def feature_selection(self):
        """
        Select the top features based on their importance in predicting the target variable.
        """
        try:
            self.logger.info("Selecting top features using mutual information...")
            # Separate the dataset into features (X) and target (y)
            X = self.df.drop(['classification'], axis=1)  # Features are all columns except 'classification'
            y = self.df['classification']  # Target variable is 'classification'

            # Use SelectKBest to select the 10 most important features
            select_k_best = SelectKBest(score_func=mutual_info_classif, k=10)
            X_selected = select_k_best.fit_transform(X, y)

            # Save the names of the selected features for later use
            self.selected_features = X.columns[select_k_best.get_support()]
            self.logger.info(f"Top features selected: {self.selected_features.tolist()}")

            # Return the reduced dataset with only the top features
            return X_selected, y
        except Exception as e:
            self.logger.error(f"Error in feature selection: {e}")
            raise

    def balance_classes(self, X, y):
        """
        Handle class imbalance using SMOTE (Synthetic Minority Over-sampling Technique).
        """
        try:
            self.logger.info("Balancing classes using SMOTE...")
            # SMOTE generates synthetic data for the minority class to balance the dataset
            smote = SMOTE(random_state=42, k_neighbors=1)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            self.logger.info(f"Classes balanced. Dataset now has {len(y_resampled)} samples.")
            return X_resampled, y_resampled
        except Exception as e:
            self.logger.error(f"Error in balancing classes: {e}")
            raise

    def scale_features(self, X_train, X_test):
        """
        Standardize the feature data to ensure all variables have similar scales.
        """
        try:
            self.logger.info("Scaling features...")
            scaler = StandardScaler()  # StandardScaler ensures mean=0 and variance=1
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            self.logger.info("Feature scaling complete. All features have been standardized.")
            return X_train, X_test
        except Exception as e:
            self.logger.error(f"Error in feature scaling: {e}")
            raise

    def train_models(self, X_train, y_train):
        """
        Train models using GridSearchCV to find the best hyperparameters.
        Log intermediate results and handle warnings effectively.
        """
        import warnings
        from sklearn.exceptions import ConvergenceWarning

        try:
            self.logger.info("Training models with GridSearchCV...")
            # Define the models and their respective hyperparameter grids
            models = {
                'Random Forest': RandomForestClassifier(random_state=42),
                'Gradient Boosting': GradientBoostingClassifier(random_state=42),
                'Neural Network': MLPClassifier(random_state=42, max_iter=300)
            }
            param_grids = {
                'Random Forest': {'n_estimators': [100, 150], 'max_depth': [4, 6, 8]},
                'Gradient Boosting': {'n_estimators': [100, 150], 'learning_rate': [0.05, 0.1]},
                'Neural Network': {'hidden_layer_sizes': [(50,), (50, 30)], 'alpha': [0.0001, 0.001]}
            }

            # Perform grid search for each model
            for name, model in models.items():
                self.logger.info(f"Starting GridSearch for {name}...")
                grid_search = GridSearchCV(model, param_grids[name], cv=5, scoring='f1_macro', verbose=2, n_jobs=-1)

                # Suppress and log convergence warnings
                with warnings.catch_warnings(record=True) as caught_warnings:
                    warnings.filterwarnings("always", category=ConvergenceWarning)
                    grid_search.fit(X_train, y_train)

                    # Log any warnings that occurred
                    for warning in caught_warnings:
                        self.logger.warning(f"Warning during training {name}: {warning.message}")

                # Save the best model and log its details
                self.best_models[name] = grid_search.best_estimator_
                self.logger.info(f"Best parameters for {name}: {grid_search.best_params_}")
                self.logger.info(f"Best estimator for {name}: {self.best_models[name]}")

            self.logger.info("Model training complete.")
        except Exception as e:
            self.logger.error(f"Error in model training: {e}")
            raise

    def evaluate_models(self, X_test, y_test):
        """
        Evaluate the trained models and generate a PDF report and a detailed text file of their performance.
        Includes classification reports, ROC curves, feature importance, and best model summary.
        """
        try:
            self.logger.info("Evaluating models...")
            pdf_path = os.path.join(self.output_dir, "model_results.pdf")
            txt_path = os.path.join(self.output_dir, "model_results.txt")
            self.logger.info(f"Saving results to {pdf_path} and {txt_path}")

            # Initialize PDF and Text File
            c = canvas.Canvas(pdf_path, pagesize=letter)
            c.setFont("Helvetica", 12)
            with open(txt_path, "w") as txt_file:
                txt_file.write("Model Evaluation Results\n")
                txt_file.write("=" * 50 + "\n\n")

                # Initialize Best Model Tracking
                best_model_name = None
                best_roc_auc = -1

                for name, model in self.best_models.items():
                    self.logger.info(f"Evaluating {name}...")
                    txt_file.write(f"Model: {name}\n")
                    y_pred = model.predict(X_test)
                    y_probs = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

                    # Classification Report
                    report = classification_report(y_test, y_pred)
                    self.logger.info(f"{name} classification report:\n{report}")
                    txt_file.write("Classification Report:\n")
                    txt_file.write(report + "\n\n")

                    # Write Classification Report to PDF
                    c.drawString(100, 750, f"{name} Classification Report:")
                    text = c.beginText(100, 730)
                    text.setFont("Helvetica", 10)
                    for line in report.split("\n"):
                        text.textLine(line)
                    c.drawText(text)
                    c.showPage()

                    # ROC Curve
                    if y_probs is not None:
                        try:
                            roc_auc = roc_auc_score(y_test, y_probs, multi_class="ovr")
                            self.logger.info(f"{name} AUC-ROC: {roc_auc:.2f}")
                            txt_file.write(f"AUC-ROC Score: {roc_auc:.2f}\n\n")

                            if roc_auc > best_roc_auc:
                                best_roc_auc = roc_auc
                                best_model_name = name

                            # Plot ROC Curve
                            plt.figure()
                            for i in range(len(np.unique(y_test))):
                                fpr, tpr, _ = roc_curve(y_test == i, y_probs[:, i])
                                plt.plot(fpr, tpr, label=f"Class {i} (AUC = {roc_auc:.2f})")
                            plt.title(f"ROC Curve - {name}")
                            plt.xlabel("False Positive Rate")
                            plt.ylabel("True Positive Rate")
                            plt.legend(loc="best")
                            roc_curve_path = os.path.join(self.output_dir, f"{name}_roc_curve.png")
                            plt.savefig(roc_curve_path)
                            plt.close()

                            # Add ROC Curve to PDF
                            c.drawString(100, 750, f"{name} ROC Curve:")
                            c.drawImage(roc_curve_path, 100, 500, width=400, height=200)
                            c.showPage()
                        except Exception as e:
                            self.logger.warning(f"Failed to generate ROC curve for {name}: {e}")

                    # Feature Importance
                    if hasattr(model, "feature_importances_"):
                        try:
                            feature_importances = model.feature_importances_
                            feature_names = self.selected_features
                            sorted_idx = np.argsort(feature_importances)

                            # Plot Feature Importance
                            plt.figure(figsize=(8, 6))
                            plt.barh(feature_names[sorted_idx], feature_importances[sorted_idx])
                            plt.title(f"Feature Importance - {name}")
                            plt.xlabel("Importance")
                            plt.ylabel("Features")
                            feature_importance_path = os.path.join(self.output_dir, f"{name}_feature_importance.png")
                            plt.savefig(feature_importance_path)
                            plt.close()

                            # Add Feature Importance Plot to PDF
                            c.drawString(100, 750, f"{name} Feature Importance:")
                            c.drawImage(feature_importance_path, 100, 500, width=400, height=200)
                            c.showPage()
                        except Exception as e:
                            self.logger.warning(f"Failed to generate feature importance plot for {name}: {e}")

                # Best Model Summary
                if best_model_name:
                    best_model = self.best_models[best_model_name]
                    c.drawString(100, 750, f"Best Model: {best_model_name}")
                    text = c.beginText(100, 730)
                    text.setFont("Helvetica", 10)
                    text.textLine(f"Best Model Parameters: {best_model.get_params()}")
                    text.textLine(f"Best Model AUC-ROC: {best_roc_auc:.2f}")
                    c.drawText(text)

                    txt_file.write(f"Best Model: {best_model_name}\n")
                    txt_file.write(f"Best Model Parameters: {best_model.get_params()}\n")
                    txt_file.write(f"Best Model AUC-ROC: {best_roc_auc:.2f}\n")

                c.save()
                self.logger.info(f"PDF saved successfully at {pdf_path}")
                self.logger.info(f"Text file saved successfully at {txt_path}")
                print(f"PDF saved at {pdf_path}")
                print(f"Text file saved at {txt_path}")

        except Exception as e:
            self.logger.error(f"Error in model evaluation: {e}")
            raise


# Initialize the pipeline and run
pipeline = CKDModelPipeline(output_dir="output")
pipeline.load_data()
pipeline.preprocess_data()
X_selected, y = pipeline.feature_selection()
X_resampled, y_resampled = pipeline.balance_classes(X_selected, y)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)
X_train, X_test = pipeline.scale_features(X_train, X_test)
pipeline.train_models(X_train, y_train)
pipeline.evaluate_models(X_test, y_test)

