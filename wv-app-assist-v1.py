import google.generativeai as genai
import json
import os
import pickle
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import re

try:
    import torch
    from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
    from safetensors import safe_open
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ Transformers dependencies not available. Install with: pip install transformers torch safetensors")

class FieldTypeCategorizer:
    """Advanced field predictor using fine-tuned DistilBERT model for field-based prediction"""
    
    def __init__(self, model_directory: str = "."):
        self.model_directory = model_directory
        self.model = None
        self.tokenizer = None
        # Common field types that can be predicted
        self.field_types = [
            "postRef", "typeId", "statusId", "privacyId", "description", "tags", 
            "assignedTo", "memId", "createdAt", "updatedAt", "title", "name", 
            "location", "date", "dueDate", "priority", "quantity", "price", 
            "contact", "email", "phone", "address", "notes", "attachments", 
            "signature", "image", "file", "category", "subcategory", "rating", 
            "review", "comment", "status", "progress", "completion", "budget", 
            "cost", "time", "duration", "startDate", "endDate", "deadline",
            "assigned", "responsible", "team", "department", "company", "client",
            "customer", "vendor", "supplier", "product", "service", "item",
            "inventory", "stock", "warehouse", "shipping", "delivery", "tracking",
            "order", "invoice", "payment", "billing", "account", "user", "role",
            "permission", "access", "security", "password", "login", "session"
        ]
        self.is_loaded = False
        self.confidence_threshold = 0.70
        
    def load_model(self) -> bool:
        """Load the fine-tuned DistilBERT model for field prediction"""
        if not TRANSFORMERS_AVAILABLE:
            print("❌ Transformers dependencies not available")
            return False
        
        # Try to load from model directory first
        if os.path.exists(self.model_directory) and os.path.isdir(self.model_directory):
            try:
                # Load the fine-tuned DistilBERT model from directory
                self.model = DistilBertForSequenceClassification.from_pretrained(self.model_directory)
                self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_directory)
                
                # Set model to evaluation mode
                self.model.eval()
                
                self.is_loaded = True
                print(f"✅ Fine-tuned DistilBERT model loaded successfully from {self.model_directory}")
                return True
                    
            except Exception as e:
                print(f"❌ Error loading DistilBERT model from directory: {e}")
        
        # Fallback: Try to load from single safetensors file
        safetensors_path = "model.safetensors"
        if os.path.exists(safetensors_path):
            try:
                print(f"📦 Attempting to load from single safetensors file: {safetensors_path}")
                
                # Load base DistilBERT model and tokenizer
                base_model_name = "distilbert-base-uncased"
                self.model = DistilBertForSequenceClassification.from_pretrained(
                    base_model_name, 
                    num_labels=len(self.field_types)
                )
                self.tokenizer = DistilBertTokenizer.from_pretrained(base_model_name)
                
                # Load custom weights from safetensors
                with safe_open(safetensors_path, framework="pt", device="cpu") as f:
                    state_dict = {}
                    for key in f.keys():
                        state_dict[key] = f.get_tensor(key)
                    
                    # Load the custom weights
                    self.model.load_state_dict(state_dict, strict=False)
                
                # Set model to evaluation mode
                self.model.eval()
                
                self.is_loaded = True
                print(f"✅ DistilBERT model loaded successfully from {safetensors_path}")
                return True
                
            except Exception as e:
                print(f"❌ Error loading from safetensors file: {e}")
        
        print(f"❌ No valid model found. Checked:")
        print(f"   - Directory: {self.model_directory}")
        print(f"   - Safetensors file: {safetensors_path}")
        return False
    
    def predict_fields(self, user_prompt: str) -> Dict[str, Any]:
        """
        Predict relevant fields using the fine-tuned DistilBERT model
        
        Args:
            user_prompt: User's app description
            
        Returns:
            Dict with predicted fields and their confidence scores
        """
        if not self.is_loaded:
            print("⚠️ Model not loaded. Using fallback field prediction.")
            return self._fallback_field_prediction(user_prompt)
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                user_prompt,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # Get model predictions using direct logits
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # Use the logits directly for field prediction
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                probabilities = probabilities.squeeze().numpy()
                
                # Get predicted fields (top fields above threshold)
                predicted_fields = []
                field_scores = {}
                
                for i, (field_name, prob) in enumerate(zip(self.field_types, probabilities)):
                    field_scores[field_name] = float(prob)
                    if prob >= self.confidence_threshold:
                        predicted_fields.append({
                            "field": field_name,
                            "confidence": float(prob),
                            "type": self._get_field_type(field_name)
                        })
                
                # Sort by confidence
                predicted_fields.sort(key=lambda x: x["confidence"], reverse=True)
                
                # Ensure we have at least some basic fields
                if not predicted_fields:
                    predicted_fields = self._get_basic_fields()
                
                return {
                    "predicted_fields": predicted_fields,
                    "all_field_scores": field_scores,
                    "model_type": "distilbert",
                    "input_text": user_prompt,
                    "total_fields": len(predicted_fields)
                }
                
        except Exception as e:
            print(f"❌ Error in DistilBERT field prediction: {e}")
            # Fallback to rule-based field prediction
            return self._fallback_field_prediction(user_prompt)
    
    def _get_field_type(self, field_name: str) -> str:
        """Determine the appropriate WizyVision field type for a given field name"""
        field_type_mapping = {
            # Text fields
            "postRef": "Text", "typeId": "Text", "title": "Text", "name": "Text",
            "description": "Paragraph", "notes": "Paragraph", "comment": "Paragraph",
            "email": "Text", "phone": "Text", "address": "Paragraph",
            
            # Dropdown fields
            "statusId": "Dropdown", "privacyId": "Dropdown", "priority": "Dropdown",
            "status": "Dropdown", "category": "Dropdown", "subcategory": "Dropdown",
            "role": "Dropdown", "permission": "Dropdown",
            
            # Date fields
            "createdAt": "Date", "updatedAt": "Date", "date": "Date", "dueDate": "Date",
            "startDate": "Date", "endDate": "Date", "deadline": "Date",
            
            # Number fields
            "quantity": "Number", "price": "Number", "budget": "Number", "cost": "Number",
            "rating": "Number", "time": "Number", "duration": "Number",
            
            # Toggle fields
            "completion": "Toggle", "progress": "Toggle",
            
            # People fields
            "assignedTo": "People", "memId": "People", "assigned": "People",
            "responsible": "People", "user": "People", "client": "People",
            "customer": "People", "vendor": "People", "supplier": "People",
            
            # Location fields
            "location": "Location",
            
            # Checkbox fields
            "tags": "Checkbox",
            
            # Signature fields
            "signature": "Signature Field",
            
            # File/Attachment fields
            "attachments": "Text", "image": "Text", "file": "Text"
        }
        
        return field_type_mapping.get(field_name, "Text")
    
    def _get_basic_fields(self) -> List[Dict[str, Any]]:
        """Return basic fields that should always be included"""
        return [
            {"field": "postRef", "confidence": 1.0, "type": "Text"},
            {"field": "typeId", "confidence": 1.0, "type": "Text"},
            {"field": "statusId", "confidence": 1.0, "type": "Dropdown"},
            {"field": "privacyId", "confidence": 1.0, "type": "Dropdown"},
            {"field": "description", "confidence": 1.0, "type": "Paragraph"},
            {"field": "tags", "confidence": 1.0, "type": "Checkbox"},
            {"field": "assignedTo", "confidence": 1.0, "type": "People"},
            {"field": "memId", "confidence": 1.0, "type": "People"},
            {"field": "createdAt", "confidence": 1.0, "type": "Date"},
            {"field": "updatedAt", "confidence": 1.0, "type": "Date"}
        ]
    
    def _fallback_field_prediction(self, user_prompt: str) -> Dict[str, Any]:
        """Fallback field prediction when model fails"""
        prompt_lower = user_prompt.lower()
        
        # Define field keywords for different app types
        field_keywords = {
            # Project/Task Management
            "project": ["title", "description", "status", "priority", "dueDate", "assignedTo", "progress"],
            "task": ["title", "description", "status", "priority", "dueDate", "assignedTo", "completion"],
            "management": ["title", "description", "status", "assignedTo", "createdAt", "updatedAt"],
            
            # Inventory/Warehouse
            "inventory": ["name", "quantity", "price", "category", "location", "supplier", "stock"],
            "warehouse": ["name", "quantity", "location", "category", "supplier", "stock"],
            "tracking": ["name", "status", "location", "date", "assignedTo", "notes"],
            
            # Customer/Client Management
            "customer": ["name", "email", "phone", "address", "company", "contact", "notes"],
            "client": ["name", "email", "phone", "address", "company", "contact", "notes"],
            "contact": ["name", "email", "phone", "address", "company", "notes"],
            
            # Healthcare/Medical
            "patient": ["name", "email", "phone", "date", "status", "notes", "doctor"],
            "medical": ["name", "date", "status", "notes", "doctor", "treatment"],
            "appointment": ["name", "date", "time", "status", "notes", "doctor"],
            
            # Education
            "student": ["name", "email", "phone", "course", "grade", "status", "teacher"],
            "course": ["title", "description", "instructor", "duration", "status", "students"],
            "education": ["title", "description", "instructor", "date", "status", "students"],
            
            # Event/Entertainment
            "event": ["title", "description", "date", "location", "attendees", "status"],
            "booking": ["title", "description", "date", "time", "location", "status"],
            "venue": ["name", "description", "location", "capacity", "contact", "status"]
        }
        
        # Find matching fields based on keywords
        predicted_fields = []
        for keyword, fields in field_keywords.items():
            if keyword in prompt_lower:
                for field in fields:
                    if field not in [pf["field"] for pf in predicted_fields]:
                        predicted_fields.append({
                            "field": field,
                            "confidence": 0.8,
                            "type": self._get_field_type(field)
                        })
        
        # Always include basic fields
        basic_fields = self._get_basic_fields()
        for basic_field in basic_fields:
            if basic_field["field"] not in [pf["field"] for pf in predicted_fields]:
                predicted_fields.append(basic_field)
        
        return {
            "predicted_fields": predicted_fields,
            "all_field_scores": {field["field"]: field["confidence"] for field in predicted_fields},
            "model_type": "fallback",
            "input_text": user_prompt,
            "total_fields": len(predicted_fields)
        }

class JSONGenerator:
    """LLM-based JSON generator that creates app structures based on categories"""
    
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        
    def generate_json(self, predicted_fields: List[Dict[str, Any]], user_prompt: str, validation_errors: List[str] = None) -> Dict[str, Any]:
        """
        Generate JSON structure based on predicted fields and user prompt
        
        Args:
            predicted_fields: List of predicted fields with their types
            user_prompt: Original user request
            validation_errors: List of validation errors to fix (if any)
            
        Returns:
            Dict[str, Any]: Generated JSON structure
        """
        
        # Build field-specific guidance
        field_guidance = []
        for field_info in predicted_fields:
            field_name = field_info["field"]
            field_type = field_info["type"]
            confidence = field_info["confidence"]
            field_guidance.append(f"- {field_name} ({field_type}, confidence: {confidence:.2f})")
        
        field_guidance_text = "\n".join(field_guidance)
        
        # Build the prompt
        base_prompt = f"""
        You are an expert app builder assistant for WizyVision. Generate a high-quality JSON app structure.
        
        User Request: "{user_prompt}"
        
        Predicted Fields to Include:
        {field_guidance_text}
        
        """
        
        if validation_errors:
            base_prompt += f"""
        Previous validation errors to fix:
        {chr(10).join(f"- {error}" for error in validation_errors)}
        
        Please address these validation issues in your new JSON structure.
        """
        
        base_prompt += """
        ## Required Standard Fields (always include):
        - postRef: string (unique identifier)
        - typeId: string (type identifier)
        - statusId: string with enum (status options)
        - privacyId: string with enum ["public", "private", "restricted"]
        - description: string (app description)
        - tags: array of strings
        - assignedTo: string (user assignment)
        - memId: string (created by user)
        - createdAt: string with date-time format
        - updatedAt: string with date-time format

        ## WizyVision Field Types:
        - Toggle: boolean
        - Checkbox: array of strings
        - Date: string with "date" or "date-time" format
        - Number: number or integer
        - Location: object with latitude:number, longitude:number, optional label:string
        - Dropdown: string enum (must include enum array)
        - Text: string (single-line)
        - Paragraph: string (multi-line)
        - People: string (userId)
        - People List: array of strings (userIds)
        - Signature Field: string

        Generate a comprehensive JSON app structure that follows WizyVision schema standards.
        Return ONLY valid JSON without markdown formatting.
        """
        
        try:
            response = self.model.generate_content(base_prompt)
            
            # Clean the response text
            json_text = response.text.strip()
            
            # Remove markdown code blocks if present
            if json_text.startswith('```json'):
                json_text = json_text[7:]
            if json_text.startswith('```'):
                json_text = json_text[3:]
            if json_text.endswith('```'):
                json_text = json_text[:-3]
            
            # Parse JSON
            app_structure = json.loads(json_text.strip())
            
            # Add metadata
            app_structure['generated_at'] = datetime.now().isoformat()
            app_structure['original_prompt'] = user_prompt
            app_structure['predicted_fields'] = predicted_fields
            app_structure['validation_attempt'] = len(validation_errors) + 1 if validation_errors else 1
            
            return app_structure
            
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing JSON response: {e}")
            return self._create_fallback_structure(predicted_fields, user_prompt, validation_errors)
            
        except Exception as e:
            print(f"❌ Error generating content: {e}")
            return self._create_fallback_structure(predicted_fields, user_prompt, validation_errors)
    
    def _create_fallback_structure(self, predicted_fields: List[Dict[str, Any]], user_prompt: str, validation_errors: List[str] = None) -> Dict[str, Any]:
        """Create a basic fallback structure"""
        return {
            "error": "Failed to generate proper JSON structure",
            "original_prompt": user_prompt,
            "predicted_fields": predicted_fields,
            "generated_at": datetime.now().isoformat(),
            "validation_attempt": len(validation_errors) + 1 if validation_errors else 1,
            "fields": {
                "postRef": {"type": "string", "x-wv-type": "Text", "description": "Unique identifier"},
                "description": {"type": "string", "x-wv-type": "Paragraph", "description": "Description"},
                "statusId": {"type": "string", "enum": ["active", "inactive"], "x-wv-type": "Dropdown", "description": "Status"}
            }
        }

class ValidationTool:
    """Schema validation tool for WizyVision JSON structures"""
    
    def __init__(self):
        self.schema_version = "2020.1"
        self.required_fields = ["postRef", "typeId", "statusId", "privacyId", "description", "tags", "assignedTo", "memId", "createdAt", "updatedAt"]
        self.valid_field_types = ["string", "number", "integer", "boolean", "array", "object"]
        self.valid_wv_types = ["Toggle", "Checkbox", "Date", "Number", "Location", "Dropdown", "Text", "Paragraph", "People", "People List", "Signature Field"]
    
    def validate_json(self, app_structure: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate the generated JSON against WizyVision schema
        
        Args:
            app_structure: The JSON structure to validate
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_errors)
        """
        errors = []
        
        try:
            # Check top-level structure
            if not isinstance(app_structure, dict):
                errors.append("Root structure must be a JSON object")
                return False, errors
            
            # Check required top-level fields
            required_top_level = ["name", "category", "fields"]
            for field in required_top_level:
                if field not in app_structure:
                    errors.append(f"Missing required top-level field: {field}")
            
            # Check fields structure
            if "fields" in app_structure:
                fields = app_structure["fields"]
                if not isinstance(fields, dict):
                    errors.append("'fields' must be a JSON object")
                else:
                    # Validate each field
                    for field_name, field_def in fields.items():
                        field_errors = self._validate_field(field_name, field_def)
                        errors.extend(field_errors)
            
            # Check for required standard fields
            if "fields" in app_structure:
                missing_required = [field for field in self.required_fields if field not in app_structure["fields"]]
                if missing_required:
                    errors.append(f"Missing required standard fields: {', '.join(missing_required)}")
            
            return len(errors) == 0, errors
            
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
            return False, errors
    
    def _validate_field(self, field_name: str, field_def: Dict[str, Any]) -> List[str]:
        """Validate individual field definition"""
        errors = []
        
        try:
            # Check if field_def is a dictionary
            if not isinstance(field_def, dict):
                errors.append(f"Field '{field_name}' must be a JSON object")
                return errors
            
            # Check required field properties
            if "type" not in field_def:
                errors.append(f"Field '{field_name}' missing 'type' property")
            else:
                if field_def["type"] not in self.valid_field_types:
                    errors.append(f"Field '{field_name}' has invalid type: {field_def['type']}")
            
            # Check x-wv-type property
            if "x-wv-type" not in field_def:
                errors.append(f"Field '{field_name}' missing 'x-wv-type' property")
            else:
                if field_def["x-wv-type"] not in self.valid_wv_types:
                    errors.append(f"Field '{field_name}' has invalid x-wv-type: {field_def['x-wv-type']}")
            
            # Validate specific field types
            if "type" in field_def and "x-wv-type" in field_def:
                type_errors = self._validate_field_type_consistency(field_name, field_def)
                errors.extend(type_errors)
            
            # Validate enum for dropdown fields
            if field_def.get("x-wv-type") == "Dropdown":
                if "enum" not in field_def:
                    errors.append(f"Dropdown field '{field_name}' must have 'enum' property")
                elif not isinstance(field_def["enum"], list):
                    errors.append(f"Field '{field_name}' enum must be an array")
            
            # Validate array items
            if field_def.get("type") == "array":
                if "items" not in field_def:
                    errors.append(f"Array field '{field_name}' must have 'items' property")
            
            # Validate location object
            if field_def.get("x-wv-type") == "Location":
                if field_def.get("type") != "object":
                    errors.append(f"Location field '{field_name}' must have type 'object'")
                elif "properties" not in field_def:
                    errors.append(f"Location field '{field_name}' must have 'properties' defined")
            
        except Exception as e:
            errors.append(f"Error validating field '{field_name}': {str(e)}")
        
        return errors
    
    def _validate_field_type_consistency(self, field_name: str, field_def: Dict[str, Any]) -> List[str]:
        """Validate consistency between type and x-wv-type"""
        errors = []
        
        field_type = field_def["type"]
        wv_type = field_def["x-wv-type"]
        
        # Type consistency rules
        type_consistency = {
            "Toggle": ["boolean"],
            "Checkbox": ["array"],
            "Date": ["string"],
            "Number": ["number", "integer"],
            "Location": ["object"],
            "Dropdown": ["string"],
            "Text": ["string"],
            "Paragraph": ["string"],
            "People": ["string"],
            "People List": ["array"],
            "Signature Field": ["string"]
        }
        
        if wv_type in type_consistency:
            if field_type not in type_consistency[wv_type]:
                errors.append(f"Field '{field_name}' type '{field_type}' is inconsistent with x-wv-type '{wv_type}'")
        
        return errors

class WizyVisionAppAssistant:
    """Main orchestrator class that integrates all tools"""
    
    def __init__(self, api_key: str):
        # Advanced field-based categorizer using fine-tuned DistilBERT
        self.categorizer = FieldTypeCategorizer()
        if self.categorizer.load_model():
            print("🚀 Fine-tuned DistilBERT model loaded successfully")
        else:
            print("⚠️ DistilBERT model not available, using fallback categorization")
        
        self.json_generator = JSONGenerator(api_key)
        self.validator = ValidationTool()
        self.max_validation_attempts = 3
        self.confidence_threshold = 0.70
        
    def process_app_request(self, user_prompt: str) -> Dict[str, Any]:
        """
        Main workflow: Categorize -> Generate -> Validate -> Iterate if needed
        
        Args:
            user_prompt: User's app description
            
        Returns:
            Dict[str, Any]: Final validated JSON structure
        """
        print("🚀 Starting WizyVision App Assistant Workflow")
        print("=" * 60)
        
        # Step 1: Predict Fields
        print("\n📊 Step 1: Predicting relevant fields...")
        
        try:
            ml_result = self.categorizer.predict_fields(user_prompt)
            predicted_fields = ml_result["predicted_fields"]
            model_type = ml_result.get("model_type", "distilbert")
            total_fields = ml_result.get("total_fields", len(predicted_fields))
            
            print(f"🎯 Predicted Fields ({model_type}): {', '.join([f['field'] for f in predicted_fields])}")
            print(f"📊 Total fields predicted: {total_fields}")
            
        except Exception as e:
            print(f"⚠️ Field prediction failed ({e}). Using fallback field prediction.")
            predicted_fields = self.categorizer._fallback_field_prediction(user_prompt)["predicted_fields"]
        
        print(f"✅ Fields determined: {', '.join([f['field'] for f in predicted_fields])}")
        
        # Step 2: Generate JSON using LLM
        print(f"\n🤖 Generating JSON structure, please wait...")
        validation_errors = []
        attempt = 1
        
        while attempt <= self.max_validation_attempts:
            print(f"\n🔄 Attempt {attempt}/{self.max_validation_attempts}")
            
            # Generate JSON
            app_structure = self.json_generator.generate_json(predicted_fields, user_prompt, validation_errors)
            
            # Step 3: Validate JSON
            print("🔍 Step 3: Validating generated JSON...")
            is_valid, errors = self.validator.validate_json(app_structure)
            
            if is_valid:
                print("✅ JSON validation successful!")
                app_structure['workflow_status'] = 'completed'
                app_structure['total_attempts'] = attempt
                return app_structure
            else:
                print(f"❌ Validation failed with {len(errors)} errors:")
                for error in errors:
                    print(f"   - {error}")
                
                validation_errors = errors
                attempt += 1
                
                if attempt <= self.max_validation_attempts:
                    print(f"\n🔄 Regenerating JSON to fix validation errors...")
                else:
                    print(f"\n⚠️ Maximum validation attempts reached. Returning best effort result.")
                    app_structure['workflow_status'] = 'completed_with_errors'
                    app_structure['validation_errors'] = errors
                    app_structure['total_attempts'] = attempt - 1
                    return app_structure
        
        return app_structure
    
    def save_result(self, app_structure: Dict[str, Any], filename: str = None) -> str:
        """Save the final result to a file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Use first predicted field or default to 'app'
            predicted_fields = app_structure.get('predicted_fields', [])
            if predicted_fields:
                first_field = predicted_fields[0]['field']
                filename = f"wv_app_{first_field.lower()}_{timestamp}.json"
            else:
                filename = f"wv_app_generated_{timestamp}.json"
        
        if not filename.endswith('.json'):
            filename += '.json'
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(app_structure, f, indent=2, ensure_ascii=False)
        
        return filename

def main():
    """Main function to run the WizyVision App Assistant"""
    print("=== WizyVision App Assistant v3.0 ===")
    print("Enhanced Workflow: Field Predictor → JSON Generator → Validation Tool")
    print("=" * 60)
    
    # Enhanced welcome message
    print("\n🎉 Welcome to WizyVision App Assistant!")
    print("I can help you build an app structure by predicting relevant fields. For example, try:")
    print("   'Create an app to track warehouse inventory'")
    print("   'Build a patient management system for clinics'")
    print("   'Design a project management app for teams'")
    print()
    
    # Get API key
    load_dotenv()
    api_key = os.getenv('GEMINI_API_KEY')
    
    if not api_key:
        api_key = input("Enter your Google AI API key: ").strip()
        if not api_key:
            print("API key is required to continue.")
            return
    
    # Initialize the assistant
    try:
        assistant = WizyVisionAppAssistant(api_key)
        print("✅ Successfully initialized WizyVision App Assistant\n")
        
    except Exception as e:
        print(f"❌ Error initializing assistant: {e}")
        return
    
    while True:
        print("\n" + "="*60)
        user_prompt = input("Describe the app you want to build (or 'quit' to exit): ").strip()
        
        if user_prompt.lower() in ['quit', 'exit', 'q']:
            print("Goodbye! 👋")
            break
        
        if not user_prompt:
            print("Please provide a valid app description.")
            continue
        
        # Process the app request through the complete workflow
        result = assistant.process_app_request(user_prompt)
        
        # Display the result
        print("\n📋 Final Generated App Structure:")
        print(json.dumps(result, indent=2))
        
        # Ask if user wants to save to file
        save_choice = input("\n💾 Save to file? (y/n): ").strip().lower()
        
        if save_choice in ['y', 'yes']:
            custom_filename = input("Enter filename (or press Enter for auto-generated): ").strip()
            filename = custom_filename if custom_filename else None
            
            try:
                saved_path = assistant.save_result(result, filename)
                print(f"✅ JSON saved to: {saved_path}")
            except Exception as e:
                print(f"❌ Error saving file: {e}")

if __name__ == "__main__":
    print("WizyVision App Assistant v3.0 - Enhanced Field Prediction System")
    print("Features:")
    print("- Field Predictor: Fine-tuned DistilBERT model for field prediction")
    print("- JSON Generator: LLM-powered structure generation")
    print("- Validation Tool: Schema compliance checking")
    print("- Field Type Mapping: Automatic WizyVision field type assignment")
    print("- Iterative refinement: Auto-fix validation errors")
    print("\nStarting application...\n")
    
    main()