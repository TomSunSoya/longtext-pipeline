"""Prompt template validation for the longtext pipeline.

This module provides validation functions for prompt template files,
checking for common issues like missing variables, malformed brackets,
and file accessibility problems.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import re


@dataclass
class ValidationIssue:
    """Represents a single validation issue found in a prompt template.
    
    Attributes:
        line_number: Line number where the issue was found (1-indexed), or None for file-level issues
        column: Column position where the issue starts (1-indexed), or None if not applicable
        issue_type: Type of issue ('missing_variable', 'unclosed_bracket', 'malformed_placeholder', 
                   'file_not_found', 'file_not_readable', 'empty_template', 'syntax_error')
        message: Human-readable description of the issue
        severity: Issue severity ('error', 'warning', 'info')
    """
    line_number: Optional[int]
    column: Optional[int]
    issue_type: str
    message: str
    severity: str = 'error'


@dataclass
class ValidationResult:
    """Represents the result of validating a prompt template.
    
    Attributes:
        is_valid: Whether the template passed all validation checks
        issues: List of validation issues found (empty if valid)
        file_path: Path to the validated file (if applicable)
        template_content: The validated template content (if validation reached that stage)
        error_count: Number of errors found
        warning_count: Number of warnings found
        info_count: Number of informational messages found
    """
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    file_path: Optional[str] = None
    template_content: Optional[str] = None
    error_count: int = 0
    warning_count: int = 0
    info_count: int = 0
    
    def __post_init__(self):
        """Calculate issue counts after initialization."""
        self.error_count = sum(1 for issue in self.issues if issue.severity == 'error')
        self.warning_count = sum(1 for issue in self.issues if issue.severity == 'warning')
        self.info_count = sum(1 for issue in self.issues if issue.severity == 'info')
        self.is_valid = self.error_count == 0


def validate_prompt_template(prompt_content: str) -> ValidationResult:
    """Validate a prompt template's content.
    
    Checks for common template issues:
    - Unclosed or unmatched {{ }} brackets
    - Malformed placeholder patterns
    - Empty or whitespace-only content
    
    Args:
        prompt_content: The template content to validate
        
    Returns:
        ValidationResult with is_valid=True if no errors found, otherwise contains issues list
    """
    issues: List[ValidationIssue] = []
    
    # Check for empty content
    if not prompt_content or not prompt_content.strip():
        issues.append(ValidationIssue(
            line_number=None,
            column=None,
            issue_type='empty_template',
            message='Template content is empty or contains only whitespace',
            severity='error'
        ))
        return ValidationResult(
            is_valid=False,
            issues=issues,
            template_content=prompt_content
        )
    
    lines = prompt_content.splitlines(keepends=False)
    
    # Track bracket depth and positions
    for line_num, line in enumerate(lines, start=1):
        # Check for unclosed opening brackets {{
        # Find all opening bracket positions
        open_bracket_positions = []
        i = 0
        while i < len(line) - 1:
            if line[i:i+2] == '{{':
                open_bracket_positions.append(i)
                i += 2
            else:
                i += 1
        
        # Find all closing bracket positions
        close_bracket_positions = []
        i = 0
        while i < len(line) - 1:
            if line[i:i+2] == '}}':
                close_bracket_positions.append(i)
                i += 2
            else:
                i += 1
        
        # Check for mismatched brackets on this line
        if len(open_bracket_positions) != len(close_bracket_positions):
            # Check if this might be a multi-line variable (we'll be lenient)
            # But flag obvious mismatches
            if len(open_bracket_positions) > len(close_bracket_positions):
                # More opens than closes - check if there's a single unclosed bracket
                unclosed_count = len(open_bracket_positions) - len(close_bracket_positions)
                if unclosed_count > 0:
                    # This might be okay for multi-line, but flag single-line obvious errors
                    # Check for patterns like "{{variable" without closing on same line
                    # when there's clearly content that should close
                    for pos in open_bracket_positions[-unclosed_count:]:
                        # Check if there's text after the opening that looks incomplete
                        after_bracket = line[pos+2:].strip()
                        if after_bracket and not after_bracket.endswith('}}'):
                            # Check if this line clearly should have a closing bracket
                            # (e.g., doesn't end with whitespace suggesting continuation)
                            if line.strip() and not line.rstrip().endswith(','):
                                issues.append(ValidationIssue(
                                    line_number=line_num,
                                    column=pos + 1,
                                    issue_type='unclosed_bracket',
                                    message=f'Unclosed opening bracket {{{{ - no matching closing bracket found',
                                    severity='error'
                                ))
        
        # Check for malformed placeholders
        # Look for single braces or mixed patterns
        single_brace_pattern = r'(?<!\{)\{(?!\{)|(?<!\})\}(?!\})'
        for match in re.finditer(single_brace_pattern, line):
            # Skip if this looks like it might be part of a valid {{ or }}
            pos = match.start()
            # Double check it's not part of a double brace
            is_single_open = line[pos] == '{' and (pos == 0 or line[pos-1] != '{') and (pos + 1 >= len(line) or line[pos+1] != '{')
            is_single_close = line[pos] == '}' and (pos == 0 or line[pos-1] != '}') and (pos + 1 >= len(line) or line[pos+1] != '}')
            
            if is_single_open or is_single_close:
                issues.append(ValidationIssue(
                    line_number=line_num,
                    column=pos + 1,
                    issue_type='malformed_placeholder',
                    message=f'Invalid single brace - use {{{{ }}}} for template variables',
                    severity='error'
                ))
    
    # Additional check: look for common patterns that suggest missing variables
    # e.g., "{{VARIABLE" without closing, or "{{ }}" empty placeholders
    empty_placeholder_pattern = r'\{\{[^\}]*\}\}'
    for match in re.finditer(r'\{\{\s*\}\}', prompt_content):
        # Find line number for this match
        line_num = prompt_content[:match.start()].count('\n') + 1
        issues.append(ValidationIssue(
            line_number=line_num,
            column=match.start() - prompt_content.split('\n')[line_num-1].find(match.group()) + 1,
            issue_type='malformed_placeholder',
            message='Empty placeholder {{ }} found - variable name required',
            severity='warning'
        ))
    
    return ValidationResult(
        is_valid=len([i for i in issues if i.severity == 'error']) == 0,
        issues=issues,
        template_content=prompt_content
    )


def validate_prompt_file(file_path: str) -> ValidationResult:
    """Validate a prompt template file.
    
    Checks:
    - File exists
    - File is readable
    - Template content is valid (delegates to validate_prompt_template)
    
    Args:
        file_path: Path to the prompt template file
        
    Returns:
        ValidationResult with is_valid=True if file exists and content is valid
    """
    issues: List[ValidationIssue] = []
    path = Path(file_path)
    
    # Check if file exists
    if not path.exists():
        issues.append(ValidationIssue(
            line_number=None,
            column=None,
            issue_type='file_not_found',
            message=f'File not found: {file_path}',
            severity='error'
        ))
        return ValidationResult(
            is_valid=False,
            issues=issues,
            file_path=file_path
        )
    
    # Check if it's actually a file (not a directory)
    if not path.is_file():
        issues.append(ValidationIssue(
            line_number=None,
            column=None,
            issue_type='file_not_readable',
            message=f'Path is not a file: {file_path}',
            severity='error'
        ))
        return ValidationResult(
            is_valid=False,
            issues=issues,
            file_path=file_path
        )
    
    # Try to read the file
    try:
        content = path.read_text(encoding='utf-8')
    except PermissionError:
        issues.append(ValidationIssue(
            line_number=None,
            column=None,
            issue_type='file_not_readable',
            message=f'Permission denied reading file: {file_path}',
            severity='error'
        ))
        return ValidationResult(
            is_valid=False,
            issues=issues,
            file_path=file_path
        )
    except Exception as e:
        issues.append(ValidationIssue(
            line_number=None,
            column=None,
            issue_type='file_not_readable',
            message=f'Error reading file: {str(e)}',
            severity='error'
        ))
        return ValidationResult(
            is_valid=False,
            issues=issues,
            file_path=file_path
        )
    
    # Validate the template content
    content_result = validate_prompt_template(content)
    
    # Combine file-level and content-level issues
    all_issues = issues + content_result.issues
    
    return ValidationResult(
        is_valid=len([i for i in all_issues if i.severity == 'error']) == 0,
        issues=all_issues,
        file_path=file_path,
        template_content=content
    )


def validate_required_variables(prompt_content: str, required_vars: List[str]) -> List[ValidationIssue]:
    """Check that specific template variables exist in the prompt content.
    
    Args:
        prompt_content: The template content to check
        required_vars: List of variable names that must be present (without braces)
        
    Returns:
        List of ValidationIssue objects for any missing variables
    """
    issues: List[ValidationIssue] = []
    
    # Find all variables in the template (pattern: {{ VARIABLE_NAME }})
    # Matches: {{variable}}, {{ variable }}, {{VARIABLE_NAME}}, etc.
    variable_pattern = r'\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}'
    found_vars = set(re.findall(variable_pattern, prompt_content))
    
    # Check each required variable
    for required_var in required_vars:
        if required_var not in found_vars:
            # Try to find similar variable names (helpful for typos)
            similar = [v for v in found_vars if required_var.lower() in v.lower() or v.lower() in required_var.lower()]
            
            message = f'Required variable {{{{{required_var}}}}} not found in template'
            if similar:
                message += f'. Did you mean: {", ".join([f"{{{{{s}}}}}" for s in similar])}?'
            
            issues.append(ValidationIssue(
                line_number=None,
                column=None,
                issue_type='missing_variable',
                message=message,
                severity='error'
            ))
    
    return issues


class TemplateValidator:
    """Public interface class for template validation.
    
    Provides a cohesive API for validating prompt template files with
    optional required variable checks.
    
    Example:
        ```python
        validator = TemplateValidator()
        
        # Basic file validation
        result = validator.validate_file('prompts/summary_general.txt')
        if not result.is_valid:
            for issue in result.issues:
                print(f"{issue.severity}: {issue.message}")
        
        # Validate with required variables
        result = validator.validate_file(
            'prompts/summary_general.txt',
            required_vars=['TEXT_TO_SUMMARIZE']
        )
        ```
    """
    
    def validate_file(self, file_path: str, required_vars: Optional[List[str]] = None) -> ValidationResult:
        """Validate a prompt template file.
        
        Args:
            file_path: Path to the prompt template file
            required_vars: Optional list of variable names that must be present
            
        Returns:
            ValidationResult with is_valid=True if all checks pass
        """
        # Validate the file
        result = validate_prompt_file(file_path)
        
        if not result.is_valid:
            return result
        
        # If required variables specified, check them
        if required_vars and result.template_content:
            var_issues = validate_required_variables(result.template_content, required_vars)
            result.issues.extend(var_issues)
            # Recalculate validity
            result.error_count = sum(1 for i in result.issues if i.severity == 'error')
            result.is_valid = result.error_count == 0
        
        return result
    
    def validate_content(self, prompt_content: str, required_vars: Optional[List[str]] = None) -> ValidationResult:
        """Validate prompt template content directly.
        
        Args:
            prompt_content: The template content to validate
            required_vars: Optional list of variable names that must be present
            
        Returns:
            ValidationResult with is_valid=True if all checks pass
        """
        result = validate_prompt_template(prompt_content)
        
        # If required variables specified, check them
        if required_vars:
            var_issues = validate_required_variables(prompt_content, required_vars)
            result.issues.extend(var_issues)
            # Recalculate validity
            result.error_count = sum(1 for i in result.issues if i.severity == 'error')
            result.is_valid = result.error_count == 0
        
        return result
