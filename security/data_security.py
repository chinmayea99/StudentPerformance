import hashlib
import os
from cryptography.fernet import Fernet
import pandas as pd

class DataSecurityManager:
    def __init__(self):
        self.encryption_key = self.load_or_generate_key()
        self.cipher_suite = Fernet(self.encryption_key)
    
    def load_or_generate_key(self):
        """Load existing encryption key or generate new one"""
        key_file = "config/encryption.key"
        if os.path.exists(key_file):
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            os.makedirs(os.path.dirname(key_file), exist_ok=True)
            with open(key_file, 'wb') as f:
                f.write(key)
            return key
    
    def anonymize_student_data(self, df, sensitive_columns):
        """Anonymize sensitive student information"""
        df_anonymized = df.copy()
        
        for column in sensitive_columns:
            if column in df_anonymized.columns:
                # Hash sensitive data
                df_anonymized[column] = df_anonymized[column].apply(
                    lambda x: hashlib.sha256(str(x).encode()).hexdigest()[:10]
                )
        
        return df_anonymized
    
    def encrypt_sensitive_data(self, data):
        """Encrypt sensitive data"""
        if isinstance(data, str):
            return self.cipher_suite.encrypt(data.encode()).decode()
        return data
    
    def decrypt_sensitive_data(self, encrypted_data):
        """Decrypt sensitive data"""
        try:
            return self.cipher_suite.decrypt(encrypted_data.encode()).decode()
        except:
            return encrypted_data
    
    def audit_data_access(self, user_id, operation, dataset, timestamp=None):
        """Log data access for audit purposes"""
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        audit_entry = {
            'timestamp': timestamp,
            'user_id': user_id,
            'operation': operation,
            'dataset': dataset,
            'ip_address': os.environ.get('USER_IP', 'localhost')
        }
        
        # Log to audit file
        audit_file = "logs/data_access_audit.log"
        os.makedirs(os.path.dirname(audit_file), exist_ok=True)
        with open(audit_file, 'a') as f:
            f.write(json.dumps(audit_entry) + '\n')
        
        return audit_entry

# Example usage
if __name__ == "__main__":
    security_manager = DataSecurityManager()
    
    # Sample student data
    sample_data = pd.DataFrame({
        'student_id': ['john.doe@email.com', 'jane.smith@email.com'],
        'name': ['John Doe', 'Jane Smith'],
        'performance_score': [85, 92]
    })
    
    # Anonymize sensitive columns
    anonymized_data = security_manager.anonymize_student_data(
        sample_data, ['student_id', 'name']
    )
    print("Anonymized Data:")
    print(anonymized_data)
