import os
import csv
import random
import string
import base64
import argparse

def get_type_from_label(label: str) -> str:
    """
    Determine the credential type based on the label/prefix used
    """
    lower_label = label.lower()
    if any(k in lower_label for k in ["passwd", "pwd", "passcode", "password"]):
        return "Password"
    elif "bearer" in lower_label:
        return "BearerToken"
    elif "token" in lower_label:
        return "Token"
    elif any(k in lower_label for k in ["api_key", "apikey", "api-key"]):
        return "APIKey"
    elif any(k in lower_label for k in ["private_key", "rsa", "ecdsa", "ssh"]):
        return "PrivateKey"
    elif any(k in lower_label for k in ["url", "ftp", "smtp", "postgresql", "sql", "mongodb"]):
        return "URL"
    else:
        return "Other"

def generate_pem_private_key() -> str:
    """Generate a mock PEM private key."""
    random_bytes = base64.b64encode(os.urandom(64)).decode()
    lines = [random_bytes[i:i+64] for i in range(0, len(random_bytes), 64)]
    return "-----BEGIN PRIVATE KEY-----\n" + "\n".join(lines) + "\n-----END PRIVATE KEY-----"

def generate_false_positive_password():
    """Generate false positive password patterns"""
    patterns = [
        f"test{random.randint(100,999)}",
        "".join(['*' * random.randint(4,8)]),
        "password123",
        f"admin{random.randint(100,999)}",
        "qwerty" + "".join(random.choices(string.digits, k=3)),
        "123456789",
        "guest" + "".join(random.choices(string.digits, k=3)),
        "default" + "".join(random.choices(string.digits, k=2)),
        "example_password",
        "mysecretpassword"
    ]
    return random.choice(patterns)

def get_separator():
    """Generate a random separator"""
    separators = [":", "=", ": ", "= ", " : ", " = "]
    return random.choice(separators)

def get_password_label():
    """Generate varied password-related labels"""
    base_labels = [
        "Password", "Passwd", "pwd", "passcode", "PassCode",
        "Password", "PWD", "PASSWD", "pass", "password_value",
        "user_password", "secret_pass", "system_password", "pass_value",
        "auth_password", "login_password", "password_key", "PassValue",
        "SecretPass", "AccountPassword", "UserPass", "AccessPass"
    ]
    return random.choice(base_labels) + get_separator()

def get_token_label(is_bearer=False):
    """Generate varied token-related labels"""
    bearer_base_labels = [
        "Token=Bearer", "Bearer", "Authorization: Bearer",
        "Authenticate: Bearer", "Auth-Token: Bearer",
        "BearerToken", "Bearer_Token", "AuthBearer",
        "Bearer-Auth", "AccessBearer", "Bearer-Token",
        "Authentication: Bearer", "BearerAuth"
    ]
    token_base_labels = [
        "Token", "auth_token", "Access-Token", "AuthToken",
        "TOKEN", "AccessToken", "Auth-Token", "TokenValue",
        "SecurityToken", "SessionToken", "UserToken",
        "ApiToken", "TokenKey", "AuthAccess", "TokenAuth",
        "AccessKey", "Authentication-Token", "TokenSecret"
    ]
    base_label = random.choice(bearer_base_labels if is_bearer else token_base_labels)
    return base_label + ('' if is_bearer else get_separator())

def get_api_key_label():
    """Generate varied API key-related labels"""
    base_labels = [
        "APIKey", "API-Key", "api_key", "ApiKey", "API_KEY",
        "X-API-Key", "api-access-key", "API_Access", "APISecret",
        "Api-Token", "API-Secret", "ApiAccess", "API_Token",
        "X-Api-Token", "ApiSecret", "API-Access", "ServiceKey",
        "API-Auth", "ApiCredential", "API_Credential"
    ]
    return random.choice(base_labels) + get_separator()

def get_url_prefix():
    """Generate varied URL prefixes"""
    prefixes = [
        "postgres://", "postgresql://", "mysql://", "mongodb://",
        "redis://", "ftp://", "sftp://", "smtp://", "ldap://",
        "cassandra://", "neo4j://", "memcached://", "elasticsearch://",
        "rabbitmq://", "mongodb+srv://", "rediss://", "mariadb://"
    ]
    return random.choice(prefixes)

def get_snippet_and_value(pattern: str) -> (str, str, str, int):
    """
    Generate a snippet code (label + credential) and the credential value only.
    Returns (snippet_code, random_cred, cred_type, is_fake)
    """
    is_fake = 0  # Default to real

    # Password generation
    if "password" in pattern.lower() or "pwd" in pattern.lower():
        label = get_password_label()
        if random.random() < 0.3:  # 30% chance of being fake
            random_cred = generate_false_positive_password()
            is_fake = 0
        else:
            length = random.randint(6, 15)
            symbols = "@_-"
            guaranteed = random.choice(symbols)
            full_alphabet = string.ascii_letters + string.digits + symbols
            remainder = random.choices(full_alphabet, k=length - 1)
            password_chars = list(guaranteed) + remainder
            random.shuffle(password_chars)
            random_cred = "".join(password_chars)
            is_fake = 1
        
        snippet_code = f"{label}{random_cred}"
        cred_type = get_type_from_label(label)

    # API Key
    elif "api" in pattern.lower():
        label = get_api_key_label()
        key_length = random.choice([32, 40])
        random_cred = "".join(random.choices("0123456789abcdef", k=key_length))
        snippet_code = f"{label}{random_cred}"
        cred_type = get_type_from_label(label)
        is_fake = 1

    # Token / Bearer token
    elif "token" in pattern.lower():
        label = get_token_label("bearer" in pattern.lower())
        
        token_length = random.randint(40, 64)
        random_cred = "".join(random.choices(string.ascii_letters + string.digits, k=token_length))
        snippet_code = f"{label}{random_cred}"
        cred_type = get_type_from_label(label)
        is_fake = 1

    # URL with credentials
    elif any(k in pattern.lower() for k in ["url", "ftp", "smtp", "postgresql"]):
        protocol = get_url_prefix().rstrip("://")
        url = f"{protocol}://{random_string(10)}:{random_string(8)}@{random_string(12)}.com"
        snippet_code = url
        random_cred = url
        cred_type = "URL"
        is_fake = 1

    # Default case
    else:
        # Skip patterns that don't match any known credential types
        return None

    return snippet_code, random_cred, cred_type, is_fake

def random_string(length):
    return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(length))

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic credential snippets.')
    parser.add_argument('--data_size', type=int, default=100, help='Number of rows to generate.')
    args = parser.parse_args()

    patterns_file = os.path.join(os.path.dirname(__file__), "regex_patterns.txt")
    with open(patterns_file, "r", encoding="utf-8") as f:
        regex_patterns = [line.strip() for line in f if line.strip() and not line.strip().startswith("//")]

    output_file = "synthetic_credentials_snippets.csv"

    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Snippet Code", "Value", "Type", "Positive"])

        generated = 0
        while generated < args.data_size:
            pattern = random.choice(regex_patterns)
            result = get_snippet_and_value(pattern)
            if result is not None:
                snippet_code, random_cred, cred_type, is_fake = result
                writer.writerow([snippet_code, random_cred, cred_type, is_fake])
                generated += 1

    print(f"File generated: {os.path.abspath(output_file)}")

if __name__ == "__main__":
    main()