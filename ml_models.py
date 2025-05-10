import pickle
import json
import logging
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from io import StringIO
import os
from utils import load_behavioral_data_to_dataframe, preprocess_behavioral_data
from models import MLModel
from app import db

logger = logging.getLogger(__name__)

class BehavioralDecisionModel:
    """Class to manage machine learning models for behavioral decision support"""
    
    def __init__(self):
        self.model = None
        self.feature_names = []
        self.target_name = ""
        self.model_type = ""
        self.model_id = None
    
    def train_decision_tree(self, data=None, target_column=None, features=None, max_depth=5):
        """
        Train a decision tree model on behavioral data
        
        Parameters:
        - data: DataFrame or None (if None, will load from database)
        - target_column: column name to predict
        - features: list of column names to use as features (if None, will use all except target)
        - max_depth: maximum depth of the decision tree
        
        Returns:
        - success: boolean indicating if training was successful
        - model_info: dictionary with model information
        """
        try:
            # Load and preprocess data if not provided
            if data is None:
                data = load_behavioral_data_to_dataframe()
                data = preprocess_behavioral_data(data)
            
            if data.empty:
                logger.error("No data available for training")
                return False, {"error": "No data available for training"}
            
            # Ensure target column exists
            if target_column is None or target_column not in data.columns:
                logger.error(f"Target column '{target_column}' not found in data")
                return False, {"error": f"Target column '{target_column}' not found in data"}
            
            # Select features
            if features is None:
                features = [col for col in data.columns if col != target_column]
            else:
                # Ensure all selected features exist in the data
                missing_features = [f for f in features if f not in data.columns]
                if missing_features:
                    logger.error(f"Features {missing_features} not found in data")
                    return False, {"error": f"Features {missing_features} not found in data"}
            
            # Prepare data
            X = data[features]
            y = data[target_column]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            # Train model
            self.model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
            self.model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # For multi-class classification, need to adjust metrics
            try:
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
            except Exception as e:
                logger.warning(f"Error calculating precision/recall/f1: {e}")
                precision = recall = f1 = None
            
            # Cross-validation
            cv_scores = cross_val_score(self.model, X, y, cv=5)
            
            # Store model info
            self.feature_names = features
            self.target_name = target_column
            self.model_type = "decision_tree"
            
            # Get feature importances
            feature_importance = dict(zip(features, self.model.feature_importances_))
            
            # Try to get text representation of the tree
            try:
                tree_text = export_text(self.model, feature_names=features)
            except Exception as e:
                logger.warning(f"Could not export tree text: {e}")
                tree_text = "Tree text export not available"
            
            # Create model info dictionary
            model_info = {
                "model_type": "decision_tree",
                "feature_names": features,
                "target_name": target_column,
                "max_depth": max_depth,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "cv_scores": cv_scores.tolist(),
                "feature_importance": feature_importance,
                "tree_text": tree_text
            }
            
            # Save model to database
            self._save_model_to_db(model_info)
            
            logger.info(f"Decision tree trained successfully with accuracy: {accuracy:.2f}")
            return True, model_info
            
        except Exception as e:
            logger.error(f"Error training decision tree: {str(e)}")
            return False, {"error": str(e)}
    
    def train_random_forest(self, data=None, target_column=None, features=None, n_estimators=100):
        """Train a random forest model on behavioral data"""
        try:
            # Load and preprocess data if not provided
            if data is None:
                data = load_behavioral_data_to_dataframe()
                data = preprocess_behavioral_data(data)
            
            if data.empty:
                logger.error("No data available for training")
                return False, {"error": "No data available for training"}
            
            # Ensure target column exists
            if target_column is None or target_column not in data.columns:
                logger.error(f"Target column '{target_column}' not found in data")
                return False, {"error": f"Target column '{target_column}' not found in data"}
            
            # Select features
            if features is None:
                features = [col for col in data.columns if col != target_column]
            else:
                # Ensure all selected features exist in the data
                missing_features = [f for f in features if f not in data.columns]
                if missing_features:
                    logger.error(f"Features {missing_features} not found in data")
                    return False, {"error": f"Features {missing_features} not found in data"}
            
            # Prepare data
            X = data[features]
            y = data[target_column]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            # Train model
            self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
            self.model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # For multi-class classification, need to adjust metrics
            try:
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
            except Exception as e:
                logger.warning(f"Error calculating precision/recall/f1: {e}")
                precision = recall = f1 = None
            
            # Cross-validation
            cv_scores = cross_val_score(self.model, X, y, cv=5)
            
            # Store model info
            self.feature_names = features
            self.target_name = target_column
            self.model_type = "random_forest"
            
            # Get feature importances
            feature_importance = dict(zip(features, self.model.feature_importances_))
            
            # Create model info dictionary
            model_info = {
                "model_type": "random_forest",
                "feature_names": features,
                "target_name": target_column,
                "n_estimators": n_estimators,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "cv_scores": cv_scores.tolist(),
                "feature_importance": feature_importance
            }
            
            # Save model to database
            self._save_model_to_db(model_info)
            
            logger.info(f"Random forest trained successfully with accuracy: {accuracy:.2f}")
            return True, model_info
            
        except Exception as e:
            logger.error(f"Error training random forest: {str(e)}")
            return False, {"error": str(e)}
    
    def predict(self, input_data):
        """
        Make a prediction using the trained model
        
        Parameters:
        - input_data: dictionary with feature values
        
        Returns:
        - prediction: the model's prediction
        - confidence: confidence score for the prediction
        - explanation: explanation of the prediction path
        """
        if self.model is None:
            logger.error("No model has been trained")
            return None, 0, "No model has been trained"
        
        try:
            # Convert input data to DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Ensure all required features are present
            missing_features = [f for f in self.feature_names if f not in input_df.columns]
            if missing_features:
                logger.error(f"Missing features: {missing_features}")
                return None, 0, f"Missing required features: {missing_features}"
            
            # Make prediction
            input_features = input_df[self.feature_names]
            prediction = self.model.predict(input_features)[0]
            
            # Get prediction probabilities if available
            confidence = 0
            if hasattr(self.model, 'predict_proba'):
                probas = self.model.predict_proba(input_features)[0]
                prediction_idx = list(self.model.classes_).index(prediction)
                confidence = probas[prediction_idx]
            
            # Generate explanation based on model type
            explanation = "Prediction path not available for this model type"
            if self.model_type == "decision_tree":
                # Get the decision path
                path = self.model.decision_path(input_features)
                node_indices = path.indices[path.indptr[0]:path.indptr[1]]
                
                # Create explanation
                explanation = "Decision path:\n"
                for i, node_idx in enumerate(node_indices):
                    if i > 0:  # Skip the root node
                        feature_idx = self.model.tree_.feature[node_indices[i-1]]
                        threshold = self.model.tree_.threshold[node_indices[i-1]]
                        feature_name = self.feature_names[feature_idx]
                        
                        # Get the actual value from input data
                        actual_value = input_features.iloc[0, feature_idx]
                        
                        # Determine if went left or right at the split
                        if actual_value <= threshold:
                            explanation += f"- {feature_name} = {actual_value:.2f} <= {threshold:.2f} (True)\n"
                        else:
                            explanation += f"- {feature_name} = {actual_value:.2f} > {threshold:.2f} (False)\n"
            
            logger.info(f"Prediction: {prediction} with confidence {confidence:.2f}")
            return prediction, confidence, explanation
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return None, 0, f"Error making prediction: {str(e)}"
    
    def load_model(self, model_id):
        """Load a model from the database"""
        try:
            model_record = MLModel.query.get(model_id)
            if not model_record:
                logger.error(f"Model with ID {model_id} not found")
                return False
            
            # The model itself is not stored in the database
            # In a production system, you'd save/load model objects
            # Here we'll just set the metadata and create a new model
            self.feature_names = json.loads(model_record.features)
            self.target_name = model_record.target
            self.model_type = model_record.model_type
            self.model_id = model_record.id
            
            # Create a simple model based on type
            if model_record.model_type == "decision_tree":
                self.model = DecisionTreeClassifier()
            elif model_record.model_type == "random_forest":
                self.model = RandomForestClassifier()
            
            logger.info(f"Model {model_id} metadata loaded")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def _save_model_to_db(self, model_info):
        """Save model metadata to the database"""
        try:
            # Create a new model record
            model_record = MLModel(
                name=f"{self.model_type.capitalize()} for {self.target_name}",
                description=f"Model to predict {self.target_name} based on behavioral data",
                model_type=self.model_type,
                features=json.dumps(self.feature_names),
                target=self.target_name,
                performance_metrics=json.dumps({
                    "accuracy": model_info.get("accuracy"),
                    "precision": model_info.get("precision"),
                    "recall": model_info.get("recall"),
                    "f1_score": model_info.get("f1_score"),
                    "cv_scores": model_info.get("cv_scores")
                })
            )
            
            db.session.add(model_record)
            db.session.commit()
            
            self.model_id = model_record.id
            logger.info(f"Model saved to database with ID {model_record.id}")
            
        except Exception as e:
            logger.error(f"Error saving model to database: {str(e)}")
            db.session.rollback()
