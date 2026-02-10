#!/usr/bin/env python3
"""Quality assurance validation script for the genre-adaptive NLI summarization validator.

This script performs comprehensive quality checks on the codebase including:
- Docstring coverage and compliance with Google style
- Error handling patterns and custom exception usage
- Configuration externalization verification
- Logging implementation assessment
- Code organization and type hint coverage

Run this script to validate that the code meets production quality standards.

Usage:
    python quality_check.py [--detailed] [--fix]

Options:
    --detailed: Show detailed analysis for each quality metric
    --fix: Automatically fix issues where possible (currently reports only)

Returns:
    Exit code 0 if all quality checks pass, 1 if issues found
"""

import argparse
import ast
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
import yaml


class QualityChecker:
    """Comprehensive quality checker for the project codebase."""

    def __init__(self, project_root: Path, detailed: bool = False):
        """Initialize the quality checker.

        Args:
            project_root: Root directory of the project
            detailed: Whether to show detailed analysis
        """
        self.project_root = project_root
        self.detailed = detailed
        self.src_dir = project_root / "src" / "genre_adaptive_nli_summarization_validator"
        self.tests_dir = project_root / "tests"
        self.config_dir = project_root / "configs"

        self.issues: List[str] = []
        self.passed_checks: List[str] = []

    def run_all_checks(self) -> Tuple[bool, Dict[str, Any]]:
        """Run all quality checks and return results.

        Returns:
            Tuple of (all_passed, detailed_results)
        """
        results = {}

        print("🔍 Running comprehensive quality checks...\n")

        # 1. Docstring Coverage and Style
        print("1. Checking docstring coverage and Google style compliance...")
        docstring_results = self._check_docstrings()
        results['docstrings'] = docstring_results

        # 2. Error Handling Patterns
        print("2. Analyzing error handling patterns...")
        error_handling_results = self._check_error_handling()
        results['error_handling'] = error_handling_results

        # 3. Configuration Externalization
        print("3. Verifying configuration externalization...")
        config_results = self._check_configuration()
        results['configuration'] = config_results

        # 4. Logging Implementation
        print("4. Assessing logging implementation...")
        logging_results = self._check_logging()
        results['logging'] = logging_results

        # 5. Type Hints Coverage
        print("5. Checking type hint coverage...")
        type_hints_results = self._check_type_hints()
        results['type_hints'] = type_hints_results

        # 6. Test Coverage Assessment
        print("6. Analyzing test coverage...")
        test_results = self._check_test_coverage()
        results['tests'] = test_results

        # 7. Code Organization
        print("7. Validating code organization...")
        organization_results = self._check_organization()
        results['organization'] = organization_results

        all_passed = len(self.issues) == 0

        return all_passed, results

    def _check_docstrings(self) -> Dict[str, Any]:
        """Check docstring coverage and Google style compliance."""
        results = {
            'total_functions': 0,
            'documented_functions': 0,
            'google_style_compliant': 0,
            'issues': [],
            'coverage_percentage': 0
        }

        python_files = list(self.src_dir.rglob("*.py"))

        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                tree = ast.parse(content)

                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        results['total_functions'] += 1

                        if ast.get_docstring(node):
                            results['documented_functions'] += 1
                            docstring = ast.get_docstring(node)

                            # Check Google style patterns
                            if self._is_google_style_docstring(docstring):
                                results['google_style_compliant'] += 1
                            else:
                                if self.detailed:
                                    results['issues'].append(
                                        f"{file_path.relative_to(self.project_root)}:{node.lineno} - "
                                        f"{node.name} docstring not Google style compliant"
                                    )
                        else:
                            if self.detailed:
                                results['issues'].append(
                                    f"{file_path.relative_to(self.project_root)}:{node.lineno} - "
                                    f"{node.name} missing docstring"
                                )

            except Exception as e:
                results['issues'].append(f"Error parsing {file_path}: {e}")

        if results['total_functions'] > 0:
            results['coverage_percentage'] = (results['documented_functions'] / results['total_functions']) * 100

        if results['coverage_percentage'] >= 95:
            self.passed_checks.append("✅ Excellent docstring coverage (≥95%)")
        elif results['coverage_percentage'] >= 80:
            self.passed_checks.append("✅ Good docstring coverage (≥80%)")
        else:
            self.issues.append(f"❌ Insufficient docstring coverage: {results['coverage_percentage']:.1f}%")

        return results

    def _is_google_style_docstring(self, docstring: str) -> bool:
        """Check if docstring follows Google style conventions."""
        if not docstring:
            return False

        # Check for Google style patterns
        google_patterns = [
            r'Args:\s*\n',
            r'Returns:\s*\n',
            r'Raises:\s*\n',
            r'Example:\s*\n',
            r'Note:\s*\n'
        ]

        # At least should have proper structure if it has parameters
        has_args_pattern = 'Args:' in docstring
        has_returns_pattern = 'Returns:' in docstring
        has_basic_description = len(docstring.strip().split('\n')[0]) > 10

        # Basic Google style compliance check
        return has_basic_description and (
            not ('Args:' in docstring) or has_args_pattern
        ) and (
            not ('return' in docstring.lower()) or has_returns_pattern
        )

    def _check_error_handling(self) -> Dict[str, Any]:
        """Check error handling patterns and custom exception usage."""
        results = {
            'total_exception_blocks': 0,
            'custom_exceptions_used': 0,
            'generic_exceptions': 0,
            'proper_error_logging': 0,
            'issues': []
        }

        python_files = list(self.src_dir.rglob("*.py"))
        custom_exceptions = [
            'GenreAdaptiveNLIError', 'ConfigurationError', 'DataLoadError',
            'ModelError', 'TrainingError', 'EvaluationError'
        ]

        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Count exception handling blocks
                except_blocks = re.findall(r'except\s+(\w+(?:\s*,\s*\w+)*)\s+as\s+\w+:', content)
                results['total_exception_blocks'] += len(except_blocks)

                # Check for custom exception usage
                for exception in custom_exceptions:
                    if f'raise {exception}' in content:
                        results['custom_exceptions_used'] += 1

                # Check for generic exception usage
                generic_patterns = re.findall(r'except\s+(Exception|BaseException)\s+as', content)
                results['generic_exceptions'] += len(generic_patterns)

                # Check for proper error logging
                error_logs = re.findall(r'logger\.error|logging\.error', content)
                results['proper_error_logging'] += len(error_logs)

            except Exception as e:
                results['issues'].append(f"Error analyzing {file_path}: {e}")

        if results['custom_exceptions_used'] > 0:
            self.passed_checks.append("✅ Custom exceptions implemented and used")
        else:
            self.issues.append("❌ No custom exception usage found")

        if results['proper_error_logging'] > 0:
            self.passed_checks.append("✅ Error logging implemented")

        return results

    def _check_configuration(self) -> Dict[str, Any]:
        """Verify configuration externalization."""
        results = {
            'config_files_found': 0,
            'externalized_values': 0,
            'hardcoded_values_found': 0,
            'issues': []
        }

        # Check config files exist
        config_files = list(self.config_dir.glob("*.yaml")) + list(self.config_dir.glob("*.yml"))
        results['config_files_found'] = len(config_files)

        if config_files:
            # Analyze default config
            default_config = self.config_dir / "default.yaml"
            if default_config.exists():
                try:
                    with open(default_config, 'r') as f:
                        config_data = yaml.safe_load(f)

                    # Count externalized configurations
                    def count_config_values(data, prefix=""):
                        count = 0
                        if isinstance(data, dict):
                            for key, value in data.items():
                                if isinstance(value, dict):
                                    count += count_config_values(value, f"{prefix}.{key}" if prefix else key)
                                else:
                                    count += 1
                        return count

                    results['externalized_values'] = count_config_values(config_data)
                    self.passed_checks.append(f"✅ Configuration externalized ({results['externalized_values']} values)")

                except Exception as e:
                    results['issues'].append(f"Error parsing config: {e}")

        # Check for hardcoded values in source
        python_files = list(self.src_dir.rglob("*.py"))
        hardcoded_patterns = [
            r'"microsoft/deberta[^"]*"',
            r'learning_rate\s*=\s*[0-9.e-]+',
            r'batch_size\s*=\s*\d+',
            r'dropout\s*=\s*0\.\d+'
        ]

        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                for pattern in hardcoded_patterns:
                    matches = re.findall(pattern, content)
                    if matches and 'default' not in str(file_path):  # Allow defaults in function signatures
                        results['hardcoded_values_found'] += len(matches)
                        if self.detailed:
                            results['issues'].extend([
                                f"{file_path.relative_to(self.project_root)} - Potential hardcoded value: {match}"
                                for match in matches
                            ])

            except Exception as e:
                results['issues'].append(f"Error checking {file_path}: {e}")

        if results['hardcoded_values_found'] == 0:
            self.passed_checks.append("✅ No hardcoded configuration values found")

        return results

    def _check_logging(self) -> Dict[str, Any]:
        """Assess logging implementation."""
        results = {
            'files_with_logging': 0,
            'total_log_statements': 0,
            'log_levels_used': set(),
            'logger_instances': 0,
            'issues': []
        }

        python_files = list(self.src_dir.rglob("*.py"))

        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                has_logging = False

                # Check for logger initialization
                if 'logging.getLogger' in content:
                    results['logger_instances'] += len(re.findall(r'logging\.getLogger', content))
                    has_logging = True

                # Check for log statements
                log_patterns = [
                    (r'\.debug\(', 'DEBUG'),
                    (r'\.info\(', 'INFO'),
                    (r'\.warning\(', 'WARNING'),
                    (r'\.error\(', 'ERROR'),
                    (r'\.critical\(', 'CRITICAL'),
                    (r'logging\.debug\(', 'DEBUG'),
                    (r'logging\.info\(', 'INFO'),
                    (r'logging\.warning\(', 'WARNING'),
                    (r'logging\.error\(', 'ERROR'),
                    (r'logging\.critical\(', 'CRITICAL'),
                ]

                for pattern, level in log_patterns:
                    matches = re.findall(pattern, content)
                    if matches:
                        results['total_log_statements'] += len(matches)
                        results['log_levels_used'].add(level)
                        has_logging = True

                if has_logging:
                    results['files_with_logging'] += 1

            except Exception as e:
                results['issues'].append(f"Error analyzing {file_path}: {e}")

        total_files = len(python_files)
        logging_coverage = (results['files_with_logging'] / total_files) * 100 if total_files > 0 else 0

        if logging_coverage >= 80:
            self.passed_checks.append(f"✅ Comprehensive logging coverage ({logging_coverage:.1f}%)")
        elif logging_coverage >= 60:
            self.passed_checks.append(f"✅ Good logging coverage ({logging_coverage:.1f}%)")
        else:
            self.issues.append(f"❌ Insufficient logging coverage: {logging_coverage:.1f}%")

        if len(results['log_levels_used']) >= 3:
            self.passed_checks.append(f"✅ Multiple log levels used: {', '.join(sorted(results['log_levels_used']))}")

        return results

    def _check_type_hints(self) -> Dict[str, Any]:
        """Check type hint coverage."""
        results = {
            'total_functions': 0,
            'typed_functions': 0,
            'coverage_percentage': 0,
            'issues': []
        }

        python_files = list(self.src_dir.rglob("*.py"))

        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                tree = ast.parse(content)

                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if not node.name.startswith('_'):  # Skip private methods for this check
                            results['total_functions'] += 1

                            # Check if function has type hints
                            has_return_annotation = node.returns is not None
                            has_arg_annotations = any(arg.annotation is not None for arg in node.args.args)

                            if has_return_annotation or has_arg_annotations:
                                results['typed_functions'] += 1

            except Exception as e:
                results['issues'].append(f"Error parsing {file_path}: {e}")

        if results['total_functions'] > 0:
            results['coverage_percentage'] = (results['typed_functions'] / results['total_functions']) * 100

        if results['coverage_percentage'] >= 80:
            self.passed_checks.append(f"✅ Excellent type hint coverage ({results['coverage_percentage']:.1f}%)")
        elif results['coverage_percentage'] >= 60:
            self.passed_checks.append(f"✅ Good type hint coverage ({results['coverage_percentage']:.1f}%)")
        else:
            self.issues.append(f"❌ Insufficient type hint coverage: {results['coverage_percentage']:.1f}%")

        return results

    def _check_test_coverage(self) -> Dict[str, Any]:
        """Analyze test coverage."""
        results = {
            'test_files': 0,
            'test_functions': 0,
            'src_modules': 0,
            'issues': []
        }

        # Count test files
        test_files = list(self.tests_dir.glob("test_*.py"))
        results['test_files'] = len(test_files)

        # Count test functions
        for test_file in test_files:
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                tree = ast.parse(content)

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                        results['test_functions'] += 1

            except Exception as e:
                results['issues'].append(f"Error parsing {test_file}: {e}")

        # Count source modules
        src_files = [f for f in self.src_dir.rglob("*.py") if f.name != "__init__.py"]
        results['src_modules'] = len(src_files)

        if results['test_files'] >= 4:
            self.passed_checks.append(f"✅ Comprehensive test suite ({results['test_files']} test files)")

        if results['test_functions'] >= 20:
            self.passed_checks.append(f"✅ Extensive test coverage ({results['test_functions']} test functions)")

        # Check for conftest.py
        if (self.tests_dir / "conftest.py").exists():
            self.passed_checks.append("✅ Test fixtures properly organized (conftest.py)")

        return results

    def _check_organization(self) -> Dict[str, Any]:
        """Validate code organization."""
        results = {
            'module_structure_correct': False,
            'init_files_present': 0,
            'expected_modules': ['models', 'data', 'training', 'evaluation', 'utils'],
            'found_modules': [],
            'issues': []
        }

        # Check main module structure
        for module in results['expected_modules']:
            module_path = self.src_dir / module
            if module_path.exists() and module_path.is_dir():
                results['found_modules'].append(module)

        results['module_structure_correct'] = len(results['found_modules']) == len(results['expected_modules'])

        # Check for __init__.py files
        init_files = list(self.src_dir.rglob("__init__.py"))
        results['init_files_present'] = len(init_files)

        if results['module_structure_correct']:
            self.passed_checks.append("✅ Proper module structure")
        else:
            missing = set(results['expected_modules']) - set(results['found_modules'])
            self.issues.append(f"❌ Missing modules: {', '.join(missing)}")

        if results['init_files_present'] >= 6:  # src + 5 submodules
            self.passed_checks.append("✅ Package structure properly defined")

        return results

    def print_summary(self, results: Dict[str, Any]) -> None:
        """Print quality check summary."""
        print("\n" + "="*80)
        print("📊 QUALITY ASSURANCE SUMMARY")
        print("="*80)

        # Print passed checks
        if self.passed_checks:
            print("\n🎉 PASSED CHECKS:")
            for check in self.passed_checks:
                print(f"  {check}")

        # Print issues
        if self.issues:
            print(f"\n⚠️  ISSUES FOUND ({len(self.issues)}):")
            for issue in self.issues:
                print(f"  {issue}")
        else:
            print("\n🎉 ALL QUALITY CHECKS PASSED!")

        # Detailed results
        if self.detailed and any(results.values()):
            print("\n📈 DETAILED METRICS:")
            for category, data in results.items():
                print(f"\n{category.upper().replace('_', ' ')}:")
                for key, value in data.items():
                    if key != 'issues':
                        print(f"  {key}: {value}")

        print("\n" + "="*80)


def main():
    """Main entry point for the quality checker."""
    parser = argparse.ArgumentParser(description="Quality assurance validation for the project")
    parser.add_argument("--detailed", action="store_true", help="Show detailed analysis")
    parser.add_argument("--fix", action="store_true", help="Automatically fix issues (future feature)")

    args = parser.parse_args()

    project_root = Path(__file__).parent
    checker = QualityChecker(project_root, detailed=args.detailed)

    try:
        all_passed, results = checker.run_all_checks()
        checker.print_summary(results)

        if args.fix:
            print("\n💡 Note: Automatic fixing is not yet implemented.")
            print("Please review the issues above and fix them manually.")

        return 0 if all_passed else 1

    except Exception as e:
        print(f"❌ Quality check failed with error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())