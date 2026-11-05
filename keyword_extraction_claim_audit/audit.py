"""Static documentation claim checks."""
import argparse
import ast
import json
import os
from pathlib import Path
import re
import sys

from . import __version__
from .phrases import GENERIC_NAMES, KEYWORD_PHRASES, NOKEYWORD_PHRASES

EXCLUDED = {'.git', '__pycache__', '.venv', 'venv', 'env', '.env', 'virtualenv', 'dist', 'build'}

def prose(text):
    lines = []
    fenced = False
    for line in text.splitlines():
        if re.match(r'^\s*```', line):
            fenced = not fenced
        elif not fenced:
            lines.append(line.replace('`', ''))
    return '\n'.join(lines)

def match_phrases(text, phrases):
    text = text.lower()
    matches = []
    for phrase in sorted(phrases, key=len, reverse=True):
        start = 0
        while True:
            pos = text.find(phrase, start)
            if pos < 0:
                break
            matches.append(phrase)
            text = text[:pos] + ' ' * len(phrase) + text[pos + len(phrase):]
            start = pos + len(phrase)
    return matches, text

def claims(text):
    negative, rest = match_phrases(prose(text), NOKEYWORD_PHRASES)
    positive, _ = match_phrases(rest, KEYWORD_PHRASES)
    return positive, negative

def evidence(source):
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return False
    nodes = list(ast.walk(tree))
    shadows = {n.name for n in nodes if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    aliases = {}
    for n in nodes:
        if isinstance(n, ast.Import):
            for a in n.names:
                aliases[a.asname or a.name.split('.')[0]] = a.name if a.asname else a.name.split('.')[0]
        elif isinstance(n, ast.ImportFrom) and n.module:
            for a in n.names:
                if a.name != '*':
                    aliases[a.asname or a.name] = n.module + '.' + a.name
    def resolve(n):
        if isinstance(n, ast.Name):
            return aliases.get(n.id, '') if n.id not in shadows else ''
        if isinstance(n, ast.Attribute):
            base = resolve(n.value)
            return base + '.' + n.attr if base and n.attr not in shadows else ''
        return ''
    constructors = {
        'rake_nltk.Rake': ('rake', 'extract_keywords_from_text'),
        'multi_rake.Rake': ('multi', 'apply'),
        'yake.KeywordExtractor': ('yake', 'extract_keywords'),
        'keybert.KeyBERT': ('keybert', 'extract_keywords'),
        'sklearn.feature_extraction.text.TfidfVectorizer': ('tfidf', None),
    }
    for group in ('unsupervised', 'supervised'):
        for name in ('TfIdf', 'TextRank', 'TopicRank', 'PositionRank', 'SingleRank', 'MultipartiteRank', 'Kea'):
            constructors['pke.' + group + '.' + name] = ('pke', 'candidate_selection')
    def kind(call):
        if not isinstance(call, ast.Call):
            return None
        return constructors.get(resolve(call.func), (None, None))[0]
    instances = {}
    shadow_instances = set()
    for n in nodes:
        if isinstance(n, (ast.Assign, ast.AnnAssign)):
            value = n.value
            targets = n.targets if isinstance(n, ast.Assign) else [n.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    if kind(value):
                        instances[target.id] = kind(value)
                    if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id in shadows:
                        shadow_instances.add(target.id)
    calls = [n for n in nodes if isinstance(n, ast.Call)]
    fit = False
    readback = False
    for n in calls:
        f = n.func
        resolved = resolve(f)
        if resolved == 'gensim.summarization.keywords':
            return True
        if isinstance(f, ast.Name):
            if f.id not in shadows and f.id in aliases and aliases[f.id].rsplit('.', 1)[-1] in GENERIC_NAMES:
                return True
        if not isinstance(f, ast.Attribute):
            continue
        base_kind = kind(f.value)
        if isinstance(f.value, ast.Name):
            base_kind = instances.get(f.value.id, base_kind)
        shadowed_base = (isinstance(f.value, ast.Name) and f.value.id in shadow_instances) or (isinstance(f.value, ast.Call) and isinstance(f.value.func, ast.Name) and f.value.func.id in shadows)
        if base_kind == 'tfidf' and f.attr in ('fit', 'fit_transform'):
            fit = True
        if f.attr == 'get_feature_names_out':
            readback = True
        if base_kind in ('rake', 'multi', 'yake', 'keybert', 'pke'):
            expected = {'rake':'extract_keywords_from_text', 'multi':'apply', 'yake':'extract_keywords', 'keybert':'extract_keywords', 'pke':'candidate_selection'}[base_kind]
            if f.attr == expected:
                return True
        if f.attr in GENERIC_NAMES and f.attr not in shadows and not shadowed_base:
            return True
    return fit and readback

def scan(repo, ignore=(), verbose=False):
    repo = Path(repo)
    if not repo.is_dir():
        raise OSError('repository is not a directory: ' + str(repo))
    files = []
    if verbose:
        print('Scan scope: ' + str(repo), file=sys.stderr)
    for root, dirs, names in os.walk(repo):
        dirs[:] = sorted(d for d in dirs if d not in EXCLUDED and not (Path(root, d, 'pyvenv.cfg').is_file()))
        for name in sorted(names):
            path = Path(root, name)
            rel = path.relative_to(repo).as_posix()
            if any(rel.startswith(prefix) for prefix in ignore):
                if verbose: print('skip ' + rel, file=sys.stderr)
                continue
            if name.endswith('.py') or rel == 'README.md' or (rel.startswith('docs/') and name.endswith('.md')) or ('tests' in path.relative_to(repo).parts and name.endswith('.md')):
                files.append((path, rel))
                if verbose: print('scan ' + rel, file=sys.stderr)
    has_evidence = False
    positives, negatives, test_positives = [], [], []
    for path, rel in files:
        text = path.read_text(encoding='utf-8')
        is_test = 'tests' in Path(rel).parts
        if path.suffix == '.py':
            if evidence(text): has_evidence = True
            if is_test:
                pos, _ = claims(text)
                if pos: test_positives.append(rel)
        elif rel == 'README.md' or rel.startswith('docs/') or is_test:
            pos, neg = claims(text)
            if is_test:
                if pos: test_positives.append(rel)
            else:
                if pos: positives.append(rel)
                if neg: negatives.append(rel)
    findings = []
    def add(verdict, severity, paths, message):
        findings.append({'verdict': verdict, 'severity': severity, 'message': message, 'paths': paths})
    if positives and not has_evidence:
        add('documented-keyword-extraction-claim-false', 'HIGH', positives, 'Documentation claims the feature without source evidence.')
    if negatives and has_evidence:
        add('documented-no-keyword-extraction-claim-false', 'HIGH', negatives, 'Documentation denies the feature despite source evidence.')
    if test_positives and not positives and not has_evidence:
        add('documented-keyword-extraction-claim-false', 'MEDIUM', test_positives, 'Only test files claim the feature without source evidence.')
    return findings

def main(argv=None):
    parser = argparse.ArgumentParser(prog='keyword-extraction-claim-audit')
    parser.add_argument('--version', action='version', version=__version__)
    sub = parser.add_subparsers(dest='command', required=True)
    audit = sub.add_parser('audit')
    audit.add_argument('repo', nargs='?', default='.')
    audit.add_argument('--json', action='store_true')
    audit.add_argument('--verbose', action='store_true')
    audit.add_argument('--ignore', action='append', default=[])
    audit.add_argument('--fail-on', choices=('high', 'medium'), default='high')
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        raise SystemExit(1 if exc.code == 2 else exc.code)
    try:
        findings = scan(args.repo, args.ignore, args.verbose)
    except (OSError, UnicodeError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
    failed = any(f['severity'] == 'HIGH' or args.fail_on == 'medium' for f in findings)
    if args.json:
        print(json.dumps(findings, indent=2))
    elif findings:
        for f in findings:
            print('{} {}: {} ({})'.format(f['severity'], f['verdict'], f['message'], ', '.join(f['paths'])))
    else:
        print('PASS')
    raise SystemExit(2 if failed else 0)
