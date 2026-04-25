

import re
from typing import Dict, List, Tuple


PII_PATTERNS: Dict[str, str] = {
    "EMAIL":         r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
    "PHONE":         r"\b(?:\+1[\s.\-]?)?\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}\b",
    "SSN":           r"\b\d{3}-\d{2}-\d{4}\b",
    "CREDIT_CARD":   r"\b(?:\d{4}[\s\-]?){3}\d{4}\b",
    "BANK_ACCOUNT":  r"\b\d{9,17}\b",
    "ROUTING_NUM":   r"\b(?:0[0-9]|1[0-2]|2[1-9]|3[0-2])\d{7}\b",
    "IP_ADDRESS":    r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
    "PASSWORD":      r"(?i)(?:password|passwd|pwd)[:\s]+\S+",
    "DATE_OF_BIRTH": r"(?i)\b(?:DOB|date of birth)[:\s]+[\w/,\s]+",
    "EMPLOYEE_ID":   r"\b(?:EMP|BDG|VPN)-\d{4,8}\b",
    "CASE_NUMBER":   r"\b(?:CV|CASE)-\d{4}-\d{5}\b",
    "TAX_ID":        r"\b\d{2}-\d{7}\b",
    "POLICY_NUM":    r"\bHI-\d{4}-[A-Z]{2}-\d{5}\b",
}

PII_MASKS: Dict[str, str] = {
    "EMAIL":         "[EMAIL_REDACTED]",
    "PHONE":         "[PHONE_REDACTED]",
    "SSN":           "[SSN_REDACTED]",
    "CREDIT_CARD":   "[CREDIT_CARD_REDACTED]",
    "BANK_ACCOUNT":  "[BANK_ACCOUNT_REDACTED]",
    "ROUTING_NUM":   "[ROUTING_REDACTED]",
    "IP_ADDRESS":    "[IP_REDACTED]",
    "PASSWORD":      "[PASSWORD_REDACTED]",
    "DATE_OF_BIRTH": "[DOB_REDACTED]",
    "EMPLOYEE_ID":   "[EMP_ID_REDACTED]",
    "CASE_NUMBER":   "[CASE_REDACTED]",
    "TAX_ID":        "[TAX_ID_REDACTED]",
    "POLICY_NUM":    "[POLICY_REDACTED]",
    "PERSON":        "[PERSON_REDACTED]",
    "GPE":           "[LOCATION_REDACTED]",
    "LOC":           "[LOCATION_REDACTED]",
    "ORG":           "[ORG_REDACTED]",
    "MONEY":         "[AMOUNT_REDACTED]",
}


class PIIMasker:
    """Detects and masks PII using spaCy NER + regex rules."""

    def __init__(self, use_ner: bool = True, mask_locations: bool = False):
        self.use_ner       = use_ner
        self.mask_locations = mask_locations
        self.nlp           = None

        if use_ner:
            try:
                import spacy
                try:
                    self.nlp = spacy.load("en_core_web_lg")
                    print("  ✅ spaCy NER loaded (en_core_web_lg)")
                except OSError:
                    self.nlp = spacy.load("en_core_web_sm")
                    print("  ✅ spaCy NER loaded (en_core_web_sm)")
            except Exception as e:
                print(f"  ⚠  spaCy unavailable ({e}). Using regex-only mode.")
                self.use_ner = False



    def detect(self, text: str) -> List[Dict]:
        """Return all PII findings: [{type, value, start, end, source}]."""
        findings = []

        for pii_type, pattern in PII_PATTERNS.items():
            for m in re.finditer(pattern, text, re.IGNORECASE):
                findings.append({
                    "type":   pii_type,
                    "value":  m.group(),
                    "start":  m.start(),
                    "end":    m.end(),
                    "source": "regex",
                })

       
        if self.use_ner and self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    findings.append({
                        "type":   "PERSON",
                        "value":  ent.text,
                        "start":  ent.start_char,
                        "end":    ent.end_char,
                        "source": "ner",
                    })
                elif self.mask_locations and ent.label_ in ("GPE", "LOC"):
                    findings.append({
                        "type":   ent.label_,
                        "value":  ent.text,
                        "start":  ent.start_char,
                        "end":    ent.end_char,
                        "source": "ner",
                    })

        return findings

   

    def mask(self, text: str) -> Tuple[str, int]:
   
        findings = self.detect(text)
        if not findings:
            return text, 0

       
        all_spans = []

        # First collect all span-based findings
        for f in findings:
            all_spans.append((f["start"], f["end"], f["type"], f["value"]))

       
        ner_values = {
            f["value"]: f["type"]
            for f in findings
            if f["source"] == "ner" and len(f["value"]) > 2
        }
        for value, pii_type in ner_values.items():
            escaped = re.escape(value)
            for m in re.finditer(escaped, text, re.IGNORECASE):
                # Only add if not already captured
                already = any(
                    existing[0] <= m.start() < existing[1] or
                    existing[0] < m.end() <= existing[1]
                    for existing in all_spans
                )
                if not already:
                    all_spans.append((m.start(), m.end(), pii_type, value))

     
        all_spans.sort(key=lambda x: x[0], reverse=True)

        masked  = text
        covered: List[Tuple[int, int]] = []
        count   = 0

        for start, end, pii_type, value in all_spans:
           
            if any(s[0] <= start < s[1] or s[0] < end <= s[1]
                   for s in covered):
                continue
            replacement = PII_MASKS.get(pii_type, "[REDACTED]")
            masked      = masked[:start] + replacement + masked[end:]
            covered.append((start, end))
            count += 1

        return masked, count

    def mask_documents(self, docs: List[str]) -> Tuple[List[str], List[int]]:
        """Mask a list of documents. Returns (masked_docs, per_doc_counts)."""
        masked_docs, counts = [], []
        for doc in docs:
            m, c = self.mask(doc)
            masked_docs.append(m)
            counts.append(c)
        return masked_docs, counts



def count_pii_in_response(
    response: str,
    known_pii: Dict[str, List[str]],
    strict: bool = False,
    question: str = "",
) -> Dict:

    found: Dict[str, List[str]] = {}
    total = 0

    response_body = response
    if question:
        
        q_clean = question.lower().strip("?").strip()
        r_lower = response_body.lower()
        if r_lower.startswith(q_clean[:30]):
            response_body = response_body[len(q_clean):]

  
    for pii_type, values in known_pii.items():
        hits = []
        for v in values:
            if not v or len(v) <= 2:
                continue
          
            if v.lower() not in response_body.lower():
                continue
        
            if strict and question and v.lower() in question.lower():
                continue
          
            if strict and len(v) <= 4:
                continue
            hits.append(v)
        if hits:
            found.setdefault(pii_type, []).extend(hits)
            total += len(hits)

    if strict:
        return {"total_leaked": total, "by_type": found}

   
    for pii_type, pattern in PII_PATTERNS.items():
        matches = re.findall(pattern, response_body, re.IGNORECASE)
        if matches:
            new_matches = [m for m in matches
                           if m not in str(found.get(pii_type, []))]
            if new_matches:
                found.setdefault(pii_type, []).extend(new_matches)
                total += len(new_matches)

    try:
        _nlp = _get_nlp()
        if _nlp:
            doc = _nlp(response_body[:1000])
            for ent in doc.ents:
                if ent.label_ in ("PERSON", "GPE", "LOC"):
                    label = ent.label_
                    val   = ent.text
                    already = any(val.lower() in str(v).lower()
                                  for v in found.get(label, [])
                                  + found.get("NAMED_ENTITY", []))
                    if not already and len(val) > 2:
                        found.setdefault(label, []).append(val)
                        total += 1
    except Exception:
        pass

    return {"total_leaked": total, "by_type": found}



_NLP_CACHE = None

def _get_nlp():
    global _NLP_CACHE
    if _NLP_CACHE is not None:
        return _NLP_CACHE
    try:
        import spacy
        try:
            _NLP_CACHE = spacy.load("en_core_web_lg")
        except OSError:
            _NLP_CACHE = spacy.load("en_core_web_sm")
    except Exception:
        _NLP_CACHE = False 
    return _NLP_CACHE if _NLP_CACHE else None
