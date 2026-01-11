import os
import re


def should_keep_print(line):
    line_lower = line.lower()
    
    error_patterns = ['error', 'failed', 'exception']
    if any(pattern in line_lower for pattern in error_patterns):
        return True
    
    test_patterns = ['test', '__main__', 'if __name__']
    for pattern in test_patterns:
        if pattern in line_lower:
            return False
    
    verbose_patterns = [
        'preprocessed:', 'entities:', 'confidence:', 'verdict:',
        'using cached', 'loading', 'initialized', 'usage:',
        'model loaded', 'added new query', 'skipping duplicate'
    ]
    if any(pattern in line_lower for pattern in verbose_patterns):
        return False
    
    return True


def simplify_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    modified = False
    new_lines = []
    
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith('print('):
            if not should_keep_print(line):
                modified = True
                continue
        
        new_lines.append(line)
    
    if modified:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        return True
    return False


def main():
    backend_path = os.path.dirname(os.path.abspath(__file__))
    
    python_files = []
    for root, dirs, files in os.walk(backend_path):
        dirs[:] = [d for d in dirs if d not in ['.venv', 'venv', '__pycache__', '.git', 'node_modules']]
        
        for file in files:
            if file.endswith('.py') and file not in ['simplify_prints.py', 'remove_comments.py', 'clean_backend.py']:
                python_files.append(os.path.join(root, file))
    
    print(f"Simplifying {len(python_files)} Python files...")
    
    simplified_count = 0
    for filepath in python_files:
        if simplify_file(filepath):
            simplified_count += 1
            rel_path = os.path.relpath(filepath, backend_path)
            print(f"Simplified: {rel_path}")
    
    print(f"\nComplete! {simplified_count} files simplified.")


if __name__ == '__main__':
    main()
