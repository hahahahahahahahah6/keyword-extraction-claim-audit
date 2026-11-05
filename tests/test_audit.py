KEY_ORACLE = ['keyword extraction', 'keyword', 'keywords', 'extracts keywords', 'extract keywords', 'keyphrase extraction', 'keyphrase', 'keyphrases', 'extracts keyphrases', 'extract keyphrases', 'keyword extractor', 'keyphrase extractor', 'keyword extraction support', 'supports keyword extraction', 'provides keyword extraction', 'performs keyword extraction', 'keyword extraction feature', 'keyword extraction capability', 'keyword extraction system', 'keyword extraction model', 'supports a keyword extractor', 'provides a keyword extractor', 'extracts keywords from text', 'extract keywords from text', 'extracts keywords from documents', 'extract keywords from documents', 'extracts key terms', 'extract key terms', 'key term extraction', 'key terms', 'automatic keyword extraction', 'auto keyword extraction', 'keyword tagging', 'keyword detection', 'top keywords', 'keyword ranking', 'ranked keywords', 'keyword scoring', 'keyword extraction pipeline', 'keyword extraction step', 'keyword extraction process', 'extracts keywords from input', 'extract keyword', 'keyword identification', 'identify keywords', 'identifies keywords', 'keyword discovery', 'discovers keywords', 'text keyword extraction', 'document keyword extraction']
NO_ORACLE = ['does not extract keywords', 'do not extract keywords', "doesn't extract keywords", "don't extract keywords", 'no keyword extraction', 'without keyword extraction', 'never extracts keywords', 'does not perform keyword extraction', 'never performs keyword extraction', 'keyword extraction not supported', 'keyword extraction unsupported', 'no keyword extraction support', 'lacks keyword extraction', 'missing keyword extraction', 'keyword extraction disabled', 'no keyword extractor', 'without a keyword extractor', 'does not provide keyword extraction', 'does not support keyword extraction', 'no automatic keyword extraction', 'without keyword extraction support', 'does not extract keyphrases', 'keyword extraction not available', 'no keyphrase extraction']
GENERIC_ORACLE = ['extract_keywords', 'extract_keyword', 'extract_keyphrases', 'extract_keyphrase', 'extract_key_terms', 'extract_keyterms', 'get_keywords', 'get_keyword', 'get_keyphrases', 'get_key_terms', 'extract_top_keywords', 'get_top_keywords', 'ranked_keywords', 'rank_keywords', 'score_keywords', 'keyword_extract', 'keyphrase_extract', 'extract_keywords_from_text', 'extract_keywords_from_document', 'find_keywords', 'identify_keywords', 'detect_keywords', 'top_keywords', 'get_ranked_keywords']

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from keyword_extraction_claim_audit.audit import claims, evidence, match_phrases, scan
from keyword_extraction_claim_audit.phrases import KEYWORD_PHRASES, NOKEYWORD_PHRASES, GENERIC_NAMES

ROOT = Path(__file__).resolve().parents[1]

class OracleTests(unittest.TestCase):
    def test_phrases(self):
        self.assertEqual(KEYWORD_PHRASES, KEY_ORACLE)
        self.assertEqual(NOKEYWORD_PHRASES, NO_ORACLE)

    def test_names(self):
        self.assertEqual(GENERIC_NAMES, GENERIC_ORACLE)

class EvidenceTests(unittest.TestCase):
    def test_positive_routes(self):
        cases = [
            'from rake_nltk import Rake as R\nr=R()\nr.extract_keywords_from_text("x")',
            'import rake_nltk as r\nr.Rake().extract_keywords_from_text("x")',
            'from multi_rake import Rake\nr=Rake()\nr.apply("x")',
            'import yake as y\nr=y.KeywordExtractor()\nr.extract_keywords("x")',
            'from keybert import KeyBERT as K\nr=K()\nr.extract_keywords("x")',
            'from sklearn.feature_extraction.text import TfidfVectorizer as T\nv=T()\nv.fit_transform([])\nv.get_feature_names_out()',
            'import gensim.summarization as s\ns.keywords("x")',
            'from gensim.summarization import keywords as k\nk("x")',
            'import pke\nr=pke.unsupervised.TfIdf()\nr.candidate_selection()',
            'from pke.unsupervised import TextRank\nTextRank().candidate_selection()',
            'obj.extract_keywords("x")',
            'from anywhere import extract_keywords as e\ne("x")',
        ]
        for case in cases:
            with self.subTest(case=case): self.assertTrue(evidence(case))

    def test_rejections(self):
        cases = [
            'import yake', 'from rake_nltk import Rake',
            'from gensim.summarization import summarize\nsummarize("x")',
            'from sumy import Summarizer\ns=Summarizer()\ns.summarize()',
            'import pke\npke.unsupervised.TextRank()',
            'from sklearn.decomposition import LatentDirichletAllocation, NMF\nLatentDirichletAllocation().fit([])\nNMF().fit_transform([])',
            'from sklearn.feature_extraction.text import TfidfVectorizer\nv=TfidfVectorizer()\nv.fit_transform([])',
            'from nltk import word_tokenize\nword_tokenize("x")\nx.tokenize()',
            'from nltk.corpus import stopwords\nstopwords.words("english")\nx.remove_stopwords()',
            'from nltk import ne_chunk\nne_chunk([])\nx.ents\npipeline("ner")',
            'pipeline("sentiment-analysis")("x")',
            'from rake_nltk import Rake\nRake()', 'from multi_rake import Rake\nRake()',
            'from yake import KeywordExtractor\nKeywordExtractor()',
            'from keybert import KeyBERT\nKeyBERT()',
            'import spacy\nspacy.load("en")',
            '# keywords\nx="keyword"\n"""keywords"""',
            'from yake import KeywordExtractor\nKeywordExtractor',
            'import subprocess\nsubprocess.run(["term-cli"])',
            'def extract_keywords(x): pass\nextract_keywords("x")',
            'from yake import KeywordExtractor\nclass KeywordExtractor: pass\nr=KeywordExtractor()\nr.extract_keywords("x")',
        ]
        for case in cases:
            with self.subTest(case=case): self.assertFalse(evidence(case))

    def test_generic_names(self):
        for name in GENERIC_ORACLE:
            with self.subTest(name=name): self.assertTrue(evidence('x.' + name + '()'))

    def test_split_modules(self):
        self.assertFalse(evidence('from multi_rake import Rake\nr=Rake()'))
        self.assertFalse(evidence('r.apply("x")'))

class MatchTests(unittest.TestCase):
    def test_negative_removal(self):
        self.assertEqual(claims('does not extract keywords'), ([], ['does not extract keywords']))
    def test_longest(self):
        self.assertEqual(match_phrases('keyword extraction system', KEYWORD_PHRASES)[0], ['keyword extraction system'])
    def test_fenced_inline(self):
        self.assertEqual(claims('```\nkeyword\n```')[0], [])
        self.assertTrue(claims('`keyword`')[0])
    def test_collision(self):
        self.assertTrue(claims('keyword arguments')[0])

class CLITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.repo = Path(self.temp.name)
    def tearDown(self): self.temp.cleanup()
    def write(self, name, text):
        p = self.repo / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding='utf-8')
    def run_cli(self, *args):
        return subprocess.run([sys.executable, '-m', 'keyword_extraction_claim_audit', *args], cwd=ROOT, capture_output=True, text=True)
    def test_good(self):
        self.write('README.md', 'keyword extraction')
        self.write('src.py', 'obj.extract_keywords("x")')
        self.assertEqual(self.run_cli('audit', str(self.repo)).returncode, 0)
    def test_bad_json(self):
        self.write('README.md', 'keyword extraction')
        p = self.run_cli('audit', str(self.repo), '--json')
        self.assertEqual(p.returncode, 2)
        f = json.loads(p.stdout)[0]
        self.assertEqual(f['verdict'], 'documented-keyword-extraction-claim-false')
        self.assertEqual(f['severity'], 'HIGH')
        self.assertTrue(f['message'])
    def test_test_only(self):
        self.write('tests/test_example.py', '"keyword extraction"')
        self.assertEqual(self.run_cli('audit', str(self.repo)).returncode, 0)
        self.assertEqual(self.run_cli('audit', str(self.repo), '--fail-on', 'medium').returncode, 2)
    def test_denial(self):
        self.write('README.md', 'does not extract keywords')
        self.write('src.py', 'obj.extract_keywords("x")')
        p = self.run_cli('audit', str(self.repo), '--json')
        self.assertEqual(p.returncode, 2)
        self.assertEqual(json.loads(p.stdout)[0]['verdict'], 'documented-no-keyword-extraction-claim-false')
    def test_no_claim(self):
        self.write('README.md', 'Hello')
        self.assertEqual(self.run_cli('audit', str(self.repo)).returncode, 0)
    def test_ignore(self):
        self.write('docs/a.md', 'keyword extraction')
        self.assertEqual(self.run_cli('audit', str(self.repo), '--ignore', 'docs/').returncode, 0)
    def test_version(self):
        p = self.run_cli('--version')
        self.assertEqual((p.returncode, p.stdout.strip()), (0, '0.1.0'))
    def test_self_readme(self):
        self.write('README.md', (ROOT / 'README.md').read_text(encoding='utf-8'))
        self.assertEqual(self.run_cli('audit', str(self.repo)).returncode, 0)

if __name__ == '__main__': unittest.main()
