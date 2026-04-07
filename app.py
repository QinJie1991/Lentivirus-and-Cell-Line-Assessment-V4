"""
慢病毒包装-细胞系评估系统

修复版本 - 解决前端模块加载错误、版本兼容性问题
"""

import streamlit as st
import requests
import json
import time
import re
import html
import logging
import sqlite3
import os
import difflib
import csv
import xml.etree.ElementTree as ET
import base64
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from io import StringIO
import pandas as pd
import zipfile
import sys

# ==================== 版本兼容性处理 ====================
def safe_rerun():
    """兼容不同 Streamlit 版本的 rerun"""
    try:
        st.rerun()
    except AttributeError:
        try:
            st.experimental_rerun()
        except:
            pass  # 如果都失败，不强制 rerun

def safe_cache_data(func):
    """兼容不同 Streamlit 版本的 cache"""
    try:
        return st.cache_data(func)
    except AttributeError:
        try:
            return st.cache(func, allow_output_mutation=True)
        except:
            return func  # 无缓存直接运行

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("lentivirus_assessment")

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="慢病毒包装-细胞系评估系统",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== HPA细胞系自动补全服务（新增）====================
class HPACellLineAutocompleteService:
    """HPA数据库细胞系名称自动补全服务 - 包含1206个细胞系的标准名称"""

    # HPA v25.0 proteinatlas.tsv 中的主要细胞系名称列表
    # 基于 "RNA cell line [Name] [nTPM]" 列提取
    HPA_CELL_LINES = [
        "A-431", "A-549", "AN3-CA", "ASC52telo", "BEWO", "BJ", "CACO-2", "Calu-6",
        "CHP-212", "Daudi", "HaCaT", "HAP1", "HBEC-3i", "HCE-2", "HCT 116", "HEK 293",
        "HeLa", "Hep G2", "HepaRG", "HL-60", "HMC-1", "HSkMC", "hTCEpi", "hTERT-HME1",
        "HUVEC", "HUVEC-Tert2", "K-562", "Karpas-707", "LNCaP", "MCF7", "MOLT-4", "NB-4",
        "NCI-H2073", "NCI-H2170", "NCI-H226", "NCI-H358", "NCI-H460", "NCI-H520",
        "NCI-H596", "NCI-H69", "NK-92", "PC-3", "REH", "RH-30", "RPMI-8226", "RT-4",
        "SCLC-21H", "SH-SY5Y", "SiHa", "SK-BR-3", "SK-MEL-30", "T-47D", "THP-1",
        "TIME", "U-138 MG", "U-2197", "U-251 MG", "U-266/70", "U-266/84", "U-698",
        "U-87 MG", "U-937", "WM-115", "WM-266-4", "A3/HL-60", "A498", "A704", "ACHN",
        "AGS", "AMO-1", "C-33A", "C3A", "Calu-1", "Calu-3", "CCRF-CEM", "CL-11",
        "COR-L23", "COR-L88", "COV318", "COV504", "Detroit 562", "DKO", "DoTc2-4510",
        "EFO-21", "EFO-27", "EM-2", "EPLC-272H", "ES-2", "ESS-1", "EW-7", "F-36P",
        "FU-OV-1", "G-401", "GAMG", "GM12878", "HBL-1", "HCC38", "HCC827", "HDLM-2",
        "HEC-1-B", "HH", "Jiyoye", "Jurkat E6.1", "KE-37", "KG-1", "KM-3", "KNS-42",
        "KPL-1", "KU-812", "L-363", "L-428", "L-540", "Loukes", "LP-1", "MHH-CALL-2",
        "MHH-CALL-3", "MHH-CALL-4", "MSTO-211H", "N-549", "NALM-6", "NCI-H1299",
        "NCI-H1395", "NCI-H1437", "NCI-H146", "NCI-H1650", "NCI-H1666", "NCI-H1694",
        "NCI-H1792", "NCI-H1836", "NCI-H1838", "NCI-H1944", "NCI-H1975", "NCI-H2009",
        "NCI-H2023", "NCI-H2030", "NCI-H2052", "NCI-H2087", "NCI-H211", "NCI-H2122",
        "NCI-H2126", "NCI-H2228", "NCI-H23", "NCI-H2342", "NCI-H2347", "NCI-H2444",
        "NCI-H2452", "NCI-H28", "NCI-H446", "NCI-H522", "NCI-H719", "NCI-H82",
        "NCI-H838", "NCI-H889", "NTERA-2", "OCI-AML2", "OCI-AML3", "OCI-LY-3",
        "OCI-LY-7", "OV-90", "PANC-1", "PC-3M-1E8", "PC-3M-2B4", "Raji", "RKO",
        "S-117", "SCLC-24H", "SCLC-5HM", "SEM", "SK-LMS-1", "SK-LU-1", "SK-N-AS",
        "SK-N-DZ", "SK-N-FI", "SK-OV-3", "SK-UT-1", "SuDHL-10", "SuDHL-16", "SuDHL-4",
        "SuDHL-5", "T-ALL-1", "T98G", "TE-10", "TE-11", "TE-12", "TE-15", "TE-441",
        "TE-617", "TE-8", "TF-1", "TK-6", "TT", "U-2 OS", "UM-UC-3", "WS-1"
    ]

    # 细胞系别名映射（支持常见书写变体）
    CELL_LINE_ALIASES = {
        "HEK 293": ["HEK293", "HEK-293", "293", "HEK293A", "HEK293T"],
        "HeLa": ["Hela", "HELA", "HeLa S3"],
        "A-549": ["A549", "A 549", "NCI-A549"],
        "Hep G2": ["HepG2", "Hep-G2", "HEPG2"],
        "MCF7": ["MCF-7", "MCF 7"],
        "U-87 MG": ["U87MG", "U-87", "U87", "U87-MG"],
        "SH-SY5Y": ["SHSY5Y", "SH SY5Y", "SY5Y"],
        "K-562": ["K562", "K 562"],
        "THP-1": ["THP1", "THP 1"],
        "PC-3": ["PC3", "PC 3"],
        "LNCaP": ["LNCAP"],
        "U-2 OS": ["U2OS", "U-2OS", "U2-OS"],
        "HCT 116": ["HCT116", "HCT-116"],
        "NCI-H226": ["NCIH226", "NCI H226", "H226"],
        "NCI-H460": ["NCIH460", "NCI H460", "H460"],
        "NCI-H520": ["NCIH520", "NCI H520", "H520"],
        "NCI-H1975": ["NCIH1975", "NCI H1975", "H1975"],
        "NCI-H1299": ["NCIH1299", "NCI H1299", "H1299"],
        "NCI-H358": ["NCIH358", "NCI H358", "H358"],
        "NCI-H23": ["NCIH23", "NCI H23", "H23"],
        "NCI-H2170": ["NCIH2170", "NCI H2170", "H2170"],
        "NCI-H2073": ["NCIH2073", "NCI H2073", "H2073"],
        "HK-2": ["HK2", "HK 2"],
        "RAW 264.7": ["RAW2647", "RAW264.7"],
        "NIH/3T3": ["NIH3T3", "NIH 3T3", "3T3"],
        "CHO-K1": ["CHOK1", "CHO K1"],
        "COS-7": ["COS7", "COS 7"],
        "BHK-21": ["BHK21", "BHK 21"],
        "PC-12": ["PC12", "PC 12"],
        "CACO-2": ["CACO2", "Caco 2"],
        "HT-29": ["HT29", "HT 29"],
        "SW-480": ["SW480", "SW 480"],
        "DU-145": ["DU145", "DU 145"],
        "Saos-2": ["Saos2", "Saos 2"],
        "MG-63": ["MG63", "MG 63"],
        "A-431": ["A431", "A 431"],
        "HaCaT": ["HACAT"],
        "BEAS-2B": ["BEAS2B", "BEAS 2B"],
        "Calu-3": ["Calu3", "Calu 3"],
        "Jurkat E6.1": ["Jurkat", "JURKAT"],
    }

    def __init__(self):
        self.search_index = {}
        self._build_search_index()

    def _build_search_index(self):
        """构建搜索索引，包括所有名称变体"""
        for cell_line in self.HPA_CELL_LINES:
            normalized = self._normalize(cell_line)
            self.search_index[normalized] = cell_line
            parts = cell_line.replace('-', ' ').replace('/', ' ').split()
            for part in parts:
                if len(part) >= 2:
                    part_norm = self._normalize(part)
                    if part_norm not in self.search_index:
                        self.search_index[part_norm] = cell_line

        for standard, aliases in self.CELL_LINE_ALIASES.items():
            for alias in aliases:
                alias_norm = self._normalize(alias)
                if alias_norm not in self.search_index:
                    hpa_name = self._find_hpa_equivalent(standard)
                    self.search_index[alias_norm] = hpa_name if hpa_name else standard

    def _find_hpa_equivalent(self, name: str):
        norm = self._normalize(name)
        for cell in self.HPA_CELL_LINES:
            if self._normalize(cell) == norm:
                return cell
        return None

    def _normalize(self, name: str) -> str:
        if not name:
            return ""
        return name.upper().replace('-', '').replace(' ', '').replace('/', '').replace('_', '')

    def get_suggestions(self, query: str, limit: int = 8) -> list:
        if not query or len(query) < 1:
            return []
        query_norm = self._normalize(query)
        if not query_norm:
            return []

        matches = []
        seen = set()

        if query_norm in self.search_index:
            cell = self.search_index[query_norm]
            matches.append({'display_name': cell, 'hpa_name': cell, 'match_type': 'exact', 'score': 100})
            seen.add(cell)

        for norm, cell in self.search_index.items():
            if norm.startswith(query_norm) and cell not in seen:
                matches.append({'display_name': cell, 'hpa_name': cell, 'match_type': 'prefix', 'score': 80 - len(norm)})
                seen.add(cell)

        for norm, cell in self.search_index.items():
            if query_norm in norm and cell not in seen:
                matches.append({'display_name': cell, 'hpa_name': cell, 'match_type': 'substring', 'score': 50})
                seen.add(cell)

        if len(query_norm) >= 3:
            for norm, cell in self.search_index.items():
                if cell not in seen:
                    sim = difflib.SequenceMatcher(None, query_norm, norm).ratio()
                    if sim > 0.6:
                        matches.append({'display_name': cell, 'hpa_name': cell, 'match_type': 'fuzzy', 'score': int(sim * 40)})
                        seen.add(cell)

        matches.sort(key=lambda x: (x['score'], x['display_name']), reverse=True)
        return matches[:limit]

    def get_exact_match(self, query: str):
        return self.search_index.get(self._normalize(query))

    def is_valid_cell_line(self, query: str) -> bool:
        return self.get_exact_match(query) is not None



# ==================== HPA基因自动补全服务（基于Gene synonym）====================
class CellLineNormalizer:
    """细胞系名称标准化、模糊匹配验证（解决书写不规范问题）"""

    STANDARD_CELL_LINES = {
        "HEK293": ["HEK 293", "HEK-293", "293", "HEK", "Human Embryonic Kidney 293", "HEK-293A", "HEK 293A"],
        "HEK293T": ["HEK 293T", "HEK-293T", "293T", "HEK293-T", "HEK 293 T", "293-T"],
        "HeLa": ["Hela", "HELA", "Hela Cell", "cervical cancer cell line", "HeLa S3"],
        "A549": ["A 549", "A-549", "lung carcinoma cell line", "A549 cell", "NCI-A549"],
        "HepG2": ["Hep G2", "Hep-G2", "hepatoma cell line", "HepG2 cell", "HEPG2"],
        "HK-2": ["HK2", "HK 2", "human kidney 2", "renal proximal tubule", "HK-2 cell", "HK 2 cell"],
        "NCI-H226": ["NCI H226", "NCIH226", "H226", "lung squamous cell carcinoma", "NCI H-226", "H-226"],
        "MCF-7": ["MCF7", "MCF 7", "breast cancer cell line", "MCF-7 cell", "MCF7 cell"],
        "U-87 MG": ["U87MG", "U-87", "U87", "glioblastoma", "U87-MG", "U 87 MG"],
        "SH-SY5Y": ["SHSY5Y", "SH SY5Y", "neuroblastoma", "SH-SY5Y cell", "SHP-SY5Y", "SY5Y"],
        "K-562": ["K562", "K 562", "chronic myeloid leukemia", "K-562 cell", "K562 cell"],
        "THP-1": ["THP1", "THP 1", "acute monocytic leukemia", "THP-1 cell", "THP1 cell"],
        "Jurkat": ["JURKAT", "T cell leukemia", "Jurkat E6.1", "Jurkat cell"],
        "RAW264.7": ["RAW 264.7", "RAW2647", "mouse macrophage", "Raw 264.7", "RAW 2647"],
        "NIH/3T3": ["NIH 3T3", "NIH3T3", "mouse fibroblast", "NIH-3T3", "3T3"],
        "CHO-K1": ["CHO K1", "CHOK1", "Chinese hamster ovary", "CHO K-1", "CHO cell"],
        "MDCK": ["Madin-Darby Canine Kidney", "dog kidney", "MDCK cell", "M.D.C.K."],
        "Vero": ["VERO", "African green monkey kidney", "Vero cell", "VERO CCL81"],
        "COS-7": ["COS7", "COS 7", "monkey kidney", "Cos-7", "COS 7 cell"],
        "BHK-21": ["BHK21", "BHK 21", "baby hamster kidney", "BHK-21 cell"],
        "PC-12": ["PC12", "PC 12", "rat adrenal pheochromocytoma", "PC-12 cell"],
        "Caco-2": ["Caco2", "Caco 2", "colorectal adenocarcinoma", "CACO-2", "Caco2 cell"],
        "HT-29": ["HT29", "HT 29", "colon cancer", "HT-29 cell", "HT29 cell"],
        "HCT116": ["HCT 116", "HCT-116", "colorectal carcinoma", "HCT-116 cell"],
        "SW480": ["SW 480", "SW-480", "colon adenocarcinoma", "SW-480 cell"],
        "DU145": ["DU 145", "DU-145", "prostate cancer", "Du145", "DU-145 cell"],
        "PC-3": ["PC3", "PC 3", "prostate cancer", "PC-3 cell", "PC3 cell"],
        "LNCaP": ["LNCaP", "lymph node carcinoma of the prostate", "LNCAP", "LNCaP cell"],
        "Saos-2": ["Saos2", "Saos 2", "osteosarcoma", "SAOS-2", "Saos2 cell"],
        "MG-63": ["MG63", "MG 63", "osteosarcoma", "MG-63 cell", "MG63 cell"],
        "U-2 OS": ["U2OS", "U-2OS", "osteosarcoma", "U2-OS", "U 2 OS"],
        "A431": ["A 431", "A-431", "epidermoid carcinoma", "A431 cell"],
        "HaCaT": ["HACAT", "human keratinocyte", "HaCaT cell", "HACAT cell"],
        "BEAS-2B": ["BEAS2B", "BEAS 2B", "bronchial epithelium", "BEAS-2B cell"],
        "Calu-3": ["Calu3", "Calu 3", "lung adenocarcinoma", "Calu-3 cell", "CALU-3"],
        "H1975": ["H-1975", "H 1975", "lung adenocarcinoma", "H1975 cell", "NCI-H1975"],
        "H1299": ["H-1299", "H 1299", "lung carcinoma", "H1299 cell", "NCI-H1299"],
        "H460": ["H-460", "H 460", "large cell lung carcinoma", "H460 cell", "NCI-H460"],
        "H358": ["H-358", "H 358", "lung adenocarcinoma", "H358 cell"],
        "H23": ["H-23", "H 23", "lung adenocarcinoma", "H23 cell", "NCI-H23"],
        "H520": ["H-520", "H 520", "lung squamous cell carcinoma", "H520 cell", "NCI-H520"],
        "H1703": ["H-1703", "H 1703", "lung squamous cell carcinoma", "H1703 cell"],
        "H2170": ["H-2170", "H 2170", "lung squamous cell carcinoma", "H2170 cell"],
    }

    CELL_TYPE_KEYWORDS = {
        'lung': ['lung', 'pulmonary', 'bronchial', 'alveolar', 'A549', 'H1975', 'H1299', 'Calu-3', 'H460', 'H358', 'H23', 'H520', 'H1703', 'H2170', 'H226', 'NCI-H'],
        'kidney': ['kidney', 'renal', 'tubule', 'HK-2', 'HEK293', 'MDCK', 'proximal'],
        'liver': ['liver', 'hepatic', 'hepatoma', 'HepG2', 'Huh7', 'Hep3B', 'hepatocyte'],
        'brain': ['brain', 'neuro', 'glioma', 'astrocyte', 'SH-SY5Y', 'U-87', 'SK-N-SH', 'neuroblastoma', 'glioblastoma'],
        'blood': ['blood', 'leukemia', 'lymphoma', 'T cell', 'B cell', 'Jurkat', 'K-562', 'THP-1', 'HL-60', 'myeloid'],
        'skin': ['skin', 'keratinocyte', 'melanoma', 'HaCaT', 'A431', 'epidermoid'],
        'colon': ['colon', 'colorectal', 'intestinal', 'Caco-2', 'HT-29', 'HCT116', 'SW480', 'rectal'],
        'prostate': ['prostate', 'LNCaP', 'PC-3', 'DU145', 'prostatic'],
        'bone': ['bone', 'osteosarcoma', 'Saos-2', 'MG-63', 'U-2 OS', 'osteoblast'],
        'fibroblast': ['fibroblast', '3T3', 'MRC-5', 'BJ', 'IMR-90'],
        'macrophage': ['macrophage', 'RAW264.7', 'THP-1', 'monocyte'],
    }

    @classmethod
    def normalize(cls, name: str) -> str:
        if not name:
            return ""
        normalized = name.strip().upper()
        normalized = ' '.join(normalized.split())
        normalized = re.sub(r'\s*-\s*', '-', normalized)
        normalized = re.sub(r'([A-Z])(\d)', r'\1-\2', normalized)
        normalized = re.sub(r'-+', '-', normalized)

        if re.match(r'^HEK-?293$', normalized) or normalized == '293':
            return 'HEK293'
        if re.match(r'^HEK-?293T$', normalized) or normalized == '293T':
            return 'HEK293T'

        if normalized.startswith('NCI') and not normalized.startswith('NCI-'):
            normalized = normalized.replace('NCI', 'NCI-', 1)

        normalized = re.sub(r'-?CELL$', '', normalized)
        return normalized

    @classmethod
    def find_best_match(cls, input_name: str) -> Tuple[Optional[str], float, List[str]]:
        if not input_name:
            return None, 0.0, []

        normalized_input = cls.normalize(input_name)
        matches = []

        for standard, aliases in cls.STANDARD_CELL_LINES.items():
            normalized_standard = cls.normalize(standard)

            if normalized_input == normalized_standard:
                return standard, 1.0, [standard]

            for alias in aliases:
                normalized_alias = cls.normalize(alias)
                if normalized_input == normalized_alias:
                    matches.append((standard, 1.0))
                    break

        if matches:
            return matches[0][0], 1.0, [m[0] for m in matches]

        best_score = 0.0
        best_matches = []

        for standard, aliases in cls.STANDARD_CELL_LINES.items():
            candidates = [standard] + aliases
            for candidate in candidates:
                score = cls._calculate_similarity(normalized_input, cls.normalize(candidate))
                if score > best_score:
                    best_score = score
                    best_matches = [(standard, score)]
                elif abs(score - best_score) < 0.01 and score > 0.5:
                    if (standard, score) not in best_matches:
                        best_matches.append((standard, score))

        if best_matches and best_score > 0.6:
            best_matches.sort(key=lambda x: x[1], reverse=True)
            top_match = best_matches[0][0]
            all_candidates = list(dict.fromkeys([m[0] for m in best_matches]))[:5]
            return top_match, best_score, all_candidates

        return None, 0.0, []

    @classmethod
    def _calculate_similarity(cls, s1: str, s2: str) -> float:
        if s1 == s2:
            return 1.0

        edit_sim = difflib.SequenceMatcher(None, s1, s2).ratio()

        substring_bonus = 0.0
        if s1 in s2 or s2 in s1:
            shorter = min(len(s1), len(s2))
            longer = max(len(s1), len(s2))
            substring_bonus = 0.2 * (shorter / longer)

        keyword_bonus = 0.0
        s1_parts = set(s1.replace('-', '').split())
        s2_parts = set(s2.replace('-', '').split())
        common = s1_parts & s2_parts
        if common:
            keyword_bonus = len(common) / max(len(s1_parts), len(s2_parts)) * 0.1

        return min(1.0, edit_sim + substring_bonus + keyword_bonus)

    @classmethod
    def get_cell_type_hint(cls, name: str) -> Optional[str]:
        name_lower = name.lower()
        matched_types = []

        for cell_type, keywords in cls.CELL_TYPE_KEYWORDS.items():
            if any(kw.lower() in name_lower for kw in keywords):
                matched_types.append(cell_type)

        return ", ".join(matched_types) if matched_types else None

    @classmethod
    def validate_and_suggest(cls, input_name: str) -> Dict:
        result = {
            'input': input_name,
            'normalized': cls.normalize(input_name),
            'is_valid': False,
            'confidence': 0.0,
            'suggested_standard': None,
            'alternatives': [],
            'cell_type': None,
            'needs_confirmation': True,
            'warning': None
        }

        if not input_name or len(input_name.strip()) < 2:
            result['warning'] = "细胞系名称过短"
            return result

        if re.search(r'[A-Z]{3,}\d{3,}', input_name) and '-' not in input_name:
            result['warning'] = f"检测到'{input_name}'可能缺少分隔符，建议格式如'NCI-H226'而非'NCIH226'"

        best_match, confidence, alternatives = cls.find_best_match(input_name)
        result['confidence'] = confidence

        if best_match:
            result['is_valid'] = True
            result['suggested_standard'] = best_match
            result['alternatives'] = alternatives[1:] if len(alternatives) > 1 else []
            result['cell_type'] = cls.get_cell_type_hint(best_match)

            if confidence >= 0.9:
                result['needs_confirmation'] = False
            elif confidence >= 0.7:
                result['needs_confirmation'] = True
            else:
                result['needs_confirmation'] = True
                result['warning'] = f"匹配度较低({confidence:.0%})，请仔细核对细胞系名称"
        else:
            result['cell_type'] = cls.get_cell_type_hint(input_name)
            result['warning'] = f"未找到匹配的细胞系'{input_name}'，将使用原始输入进行检索"

        return result

# ==================== AI分析客户端 ====================
class AIAnalysisClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"

        self.CELL_CULTURE_DIFFICULTY_CHECKLIST = {
            "培养基特殊要求": [
                "无血清培养基（如UltraCULTURE、SFM）",
                "需要添加特殊血清（如马血清、新生牛血清）",
                "需要添加生长因子（如EGF、bFGF、HGF、VEGF等）",
                "需要添加激素（如胰岛素、氢化可的松、地塞米松）",
                "需要添加微量元素（如硒酸钠、转铁蛋白）",
                "培养基价格>200元/500ml（如StemPro、mTeSR）",
                "需要定制化培养基（如添加特定脂肪酸、胆固醇）",
                "需要条件培养基（收集其他细胞上清）"
            ],
            "基质/包被要求": [
                "需要Matrigel包被（价格昂贵，4°C操作）",
                "需要胶原蛋白包被（I型/IV型）",
                "需要纤连蛋白（Fibronectin）包被",
                "需要明胶包被（0.1% Gelatin）",
                "需要多聚赖氨酸（Poly-L-lysine）包被",
                "需要饲养层细胞（Feeder layer，如MEF、STO）",
                "需要低吸附培养板（Ultra-Low Attachment）",
                "需要专用培养装置（如Spinner flask、旋转瓶）"
            ],
            "气体/环境敏感": [
                "需要低氧培养（5%或更低O₂，非常规CO₂培养箱）",
                "对pH极度敏感（需每日观察颜色变化）",
                "需要持续震荡/搅拌（贴壁培养会死亡）",
                "对CO₂浓度敏感（需精确控制5-10%）",
                "对温度波动敏感（不能长时间开门）"
            ],
            "操作复杂度": [
                "密度极度敏感（太稀死，太密也死，窗口窄）",
                "对机械力敏感（吹打、离心即死）",
                "需要每天换液/传代（周末无法休息）",
                "需要半量换液（不能全换，否则死亡）",
                "需要频繁观察/拍照（每日至少2次）",
                "胰酶消化时间窗口窄（<2分钟或>10分钟）",
                "容易成团成簇（需频繁吹打分散）",
                "容易自发分化（需持续加因子维持）"
            ],
            "时间/成本": [
                "倍增时间>48小时（生长极其缓慢）",
                "倍增时间<12小时（需每天传代，累死人）",
                "原代细胞（无法扩增，需反复取材）",
                "有限细胞系（传代次数受限，如<10代）",
                "需要特定批次血清（不同批次差异大）",
                "支原体敏感（一旦污染立即死亡，且形态无明显变化）"
            ]
        }

    def analyze_antiviral_evidence(self, gene_name: str, title: str, abstract: str) -> Dict:
        """
        分析文献中是否报道了基因的抗病毒功能。
        严格基于提供的文献内容，禁止推测。
        """
        if not self.api_key:
            return {'is_antiviral': False, 'confidence': 0, 'mechanism': '', 'reasoning': '未配置API'}

        try:
            prompt = f"""请【严格基于以下文献内容】判断基因"{gene_name}"是否具有抗病毒功能，并进行语义分析和归纳总结。

【重要警告】你绝对不能进行推测：
1. 只能基于提供的文献标题和摘要判断
2. 如果文献未明确提及抗病毒功能，必须返回false
3. 不要基于基因名称或一般知识进行推断

文献标题：{title}
文献摘要：{abstract}

请按以下JSON格式回答（只返回JSON）：
{{
    "is_antiviral": true/false,
    "confidence": 0.0-1.0,
    "mechanism": "文献明确描述的抗病毒机制，如无则留空",
    "reasoning": "引用文献中的具体描述作为判断依据",
    
    "semantic_analysis": {{
        "evidence_strength": "基于文献的语义证据强度评估（Strong/Moderate/Weak/None）",
        "functional_context": "文献中描述的功能上下文",
        "implied_roles": ["文献暗示的可能作用（但不编造）"],
        "terminology_signals": "文献用语的信号分析"
    }},
    
    "inductive_summary": {{
        "core_claim": "文献的核心主张总结",
        "supporting_evidence": "支持抗病毒功能的证据",
        "limitations": "文献中提到的局限性或条件",
        "research_context": "研究背景对结论的影响"
    }}
}}

严格标准：
1. is_antiviral=true：文献【明确】提到该基因能抑制病毒复制、增强抗病毒免疫等
2. is_antiviral=false：文献未提及抗病毒功能，或仅提及其他功能
3. confidence：0.0-1.0，基于证据明确程度
4. mechanism：必须是文献中明确描述的，禁止推测
5. reasoning：必须引用文献原文或准确概括文献内容
6. 【语义分析】深度理解文献的表述方式和隐含信息
7. 【归纳总结】提炼文献的核心发现和证据结构"""

            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            payload = {
                'model': 'qwen-turbo',
                'input': {
                    'messages': [
                        {'role': 'system', 'content': '你是一个严谨的生物医学文献分析助手。你只基于提供的文献内容进行判断，绝不进行推测或引申。'},
                        {'role': 'user', 'content': prompt}
                    ]
                },
                'parameters': {
                    'result_format': 'message',
                    'max_tokens': 500,
                    'temperature': 0.1
                }
            }

            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()

            result = response.json()
            content = result.get('output', {}).get('choices', [{}])[0].get('message', {}).get('content', '')

            try:
                content_clean = content.replace('```json', '').replace('```', '').strip()
                analysis = json.loads(content_clean)
                return {
                    'is_antiviral': analysis.get('is_antiviral', False),
                    'confidence': float(analysis.get('confidence', 0)),
                    'mechanism': analysis.get('mechanism', ''),
                    'reasoning': analysis.get('reasoning', '')
                }
            except json.JSONDecodeError:
                # 降级处理：使用关键词匹配
                is_antiviral = any(kw in (title + abstract).lower() for kw in
                    ['antiviral', 'virus', 'interferon', 'ifitm', 'innate immunity'])
                return {
                    'is_antiviral': is_antiviral,
                    'confidence': 0.3 if is_antiviral else 0,  # 降低置信度
                    'mechanism': '解析失败，使用关键词匹配（仅供参考）',
                    'reasoning': 'API返回格式异常，使用关键词匹配降级处理'
                }

        except Exception as e:
            return {
                'is_antiviral': False,
                'confidence': 0,
                'mechanism': '',
                'reasoning': f'API调用失败: {str(e)}'
            }

    def analyze_gene_function_comprehensive(self, gene_name: str, gene_description: str,
                                          papers_oe: List[Dict], papers_kd: List[Dict],
                                          papers_ko: List[Dict], papers_general: List[Dict]) -> Dict:
        if not self.api_key:
            return {'error': '未配置AI API', 'note': '请在侧边栏配置API Key或在secrets中设置DASHSCOPE_API_KEY'}

        def format_papers(papers, label):
            if not papers:
                return f"\n{label}文献：无相关文献\n"
            text = f"\n{label}文献：\n"
            for i, p in enumerate(papers[:5], 1):
                text += f"{i}. {p.get('title', '')} - {p.get('abstract', '')[:300]}... PMID:{p.get('pmid', 'N/A')}\n"
            return text

        literature_text = ""
        literature_text += format_papers(papers_general, "基因功能相关")
        literature_text += format_papers(papers_oe, "过表达")
        literature_text += format_papers(papers_kd, "敲低/敲除")
        literature_text += format_papers(papers_ko, "敲除")

        # 【严格模式】如果没有文献，返回空结果
        total_papers = len(papers_general) + len(papers_oe) + len(papers_kd) + len(papers_ko)
        if total_papers == 0:
            return {
                'protein_function': {
                    'category': '未检索到文献',
                    'domains': '未检索到文献',
                    'pathways': '未检索到文献',
                    'cellular_location': '未检索到文献',
                    'tissue_expression': '未检索到文献'
                },
                'overexpression': {'cell_models': [], 'animal_models': [], 'summary': '无相关文献'},
                'knockdown': {'cell_models': [], 'summary': '无相关文献'},
                'knockout': {'cell_models': [], 'animal_models': [], 'summary': '无相关文献'},
                'disease_relevance': {
                    'cancer': '未检索到文献',
                    'other_diseases': '未检索到文献',
                    'therapeutic_potential': '未检索到文献'
                },
                'key_references': [],
                'experimental_notes': '未检索到该基因的相关文献，无法提供功能总结。建议查阅NCBI Gene数据库或PubMed获取最新信息。',
                'data_source_note': '未找到相关文献，AI未进行推测'
            }

        prompt = f"""作为分子生物学和遗传学专家，请【严格基于以下提供的文献】总结基因"{gene_name}"的功能及实验模型数据。

【重要警告】你绝对不能进行推测或编造：
1. 只能基于提供的文献回答问题
2. 如果文献未提及某项信息，该字段必须标注"文献未提供"
3. 禁止基于基因名称或一般知识进行推断
4. 每个具体数据必须标注文献来源（PMID）

提供的文献：
{literature_text}

请按以下JSON格式提供结构化总结（只返回JSON）：
{{
    "protein_function": {{
        "category": "文献明确描述的功能类别 + [PMID:XXXX]",
        "domains": "文献明确描述的结构域 + [PMID:XXXX]",
        "pathways": "文献明确描述的通路 + [PMID:XXXX]",
        "cellular_location": "文献明确描述的亚细胞定位 + [PMID:XXXX]",
        "tissue_expression": "文献明确描述的组织表达 + [PMID:XXXX]"
    }},
    "overexpression": {{
        "cell_models": [
            {{
                "cell_line": "文献明确报道的细胞系",
                "phenotype": "文献明确报道的表型",
                "mechanism": "文献明确描述的机制",
                "reference": "PMID:XXXX（文献标题）"
            }}
        ],
        "animal_models": [
            {{
                "model": "文献明确报道的动物模型",
                "phenotype": "文献明确报道的表型",
                "reference": "PMID:XXXX（文献标题）"
            }}
        ],
        "summary": "基于文献的过表达效应总结"
    }},
    "knockdown": {{
        "cell_models": [
            {{
                "cell_line": "文献明确报道的细胞系",
                "method": "文献明确报道的敲低方法",
                "phenotype": "文献明确报道的表型",
                "reference": "PMID:XXXX（文献标题）"
            }}
        ],
        "summary": "基于文献的敲低效应总结"
    }},
    "knockout": {{
        "cell_models": [
            {{
                "cell_line": "文献明确报道的细胞系",
                "method": "文献明确报道的敲除方法",
                "phenotype": "文献明确报道的表型",
                "viability": "文献明确报道的细胞活力影响",
                "reference": "PMID:XXXX（文献标题）"
            }}
        ],
        "animal_models": [
            {{
                "model": "文献明确报道的动物模型",
                "phenotype": "文献明确报道的表型",
                "lethality": "文献明确报道的致死性",
                "reference": "PMID:XXXX（文献标题）"
            }}
        ],
        "summary": "基于文献的敲除效应总结"
    }},
    "disease_relevance": {{
        "cancer": "文献明确描述的肿瘤作用 + [PMID:XXXX]",
        "other_diseases": "文献明确描述的其他疾病 + [PMID:XXXX]",
        "therapeutic_potential": "文献明确描述的治疗潜力 + [PMID:XXXX]"
    }},
    "key_references": [
        "PMID:XXXX（文献标题）"
    ],
    "experimental_notes": "基于文献的实验设计建议",
    "data_source_note": "基于X篇文献分析，所有信息均有文献支持",
    
    "semantic_analysis": {{
        "gene_character": "基于文献的基因特性语义描述（如：促癌基因/抑癌基因/必需基因等）",
        "functional_complexity": "功能复杂度的语义评估",
        "experimental_challenges": ["文献中暗示的实验挑战"],
        "safety_considerations": "基于文献的安全考虑"
    }},
    
    "inductive_summary": {{
        "functional_consensus": "多篇文献对该基因功能的共识",
        "phenotypic_patterns": "表型模式总结（如：过表达总是促进增殖）",
        "context_dependent_effects": "上下文依赖性效应（如：细胞类型特异性）",
        "knowledge_conflicts": "文献间的矛盾或不一致之处",
        "research_gaps": "研究空白和未来方向"
    }}
}}

严格要求：
1. 【禁止推测】文献未明确报道的信息必须标注"文献未提供"
2. 【必须标注来源】每个具体描述后必须标注[PMID:XXXX]
3. 【禁止常识推断】即使是常见基因（如p53、GAPDH），没有文献支持不得输出
4. 如果某类文献缺失（如无敲除文献），该部分必须返回空数组或标注"无相关文献"
5. key_references只能列出上述提供的真实文献，必须包含准确PMID
6. 【语义分析】深度理解文献中的功能描述和隐含信息
7. 【归纳总结】整合多篇文献的发现，找出功能模式、共识和矛盾"""

        try:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            payload = {
                'model': 'qwen-turbo',
                'input': {
                    'messages': [
                        {'role': 'system', 'content': '你是分子生物学专家，精通基因功能注释和表型分析。你只能基于用户提供的文献进行总结，绝对禁止编造任何文献、PMID、作者或期刊信息。reference字段必须严格使用格式："第一作者 et al., 年份, 期刊缩写, PMID:数字"。如果信息不确定，明确标注"未见报道"。'},
                        {'role': 'user', 'content': prompt}
                    ]
                },
                'parameters': {
                    'result_format': 'message',
                    'max_tokens': 3000,
                    'temperature': 0.2
                }
            }

            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()

            result = response.json()
            content = result.get('output', {}).get('choices', [{}])[0].get('message', {}).get('content', '')

            try:
                content_clean = content.replace('```json', '').replace('```', '').strip()
                return json.loads(content_clean)
            except json.JSONDecodeError:
                return {
                    'error': 'AI返回格式异常',
                    'raw_response': content[:1000],
                    'note': 'AI未能返回有效JSON格式'
                }

        except Exception as e:
            return {'error': str(e), 'note': 'API调用失败，请检查网络或API Key有效性'}

    def design_rnai_sequences(self, gene_name: str, gene_id: str = "",
                             gene_description: str = "") -> Dict:
        """从PMC全文文献和专利数据库中检索siRNA/shRNA序列（严格仅使用文献Materials and Methods中报道的序列，禁止AI生成或推测）"""
        try:
            import re
            
            # 设置NCBI API参数
            email, api_key, error = APIConfig.get_ncbi_credentials()
            if error:
                return {'error': 'NCBI API未配置', 'sequences': [], 'source': 'literature'}
            
            base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
            
            def ncbi_request(endpoint, params):
                """发送NCBI API请求"""
                params['tool'] = 'LentivirusAssessment'
                params['email'] = email
                if api_key:
                    params['api_key'] = api_key
                try:
                    ncbi_limiter.wait()
                    response = requests.get(f"{base_url}/{endpoint}", params=params, timeout=30)
                    response.raise_for_status()
                    if params.get('retmode') == 'json':
                        return response.json()
                    return response.text
                except Exception as e:
                    logger.error(f"NCBI请求失败: {e}")
                    return None
            
            # 搜索PMC（PubMed Central）全文数据库
            search_terms = [
                f'({gene_name}[Title/Abstract]) AND (siRNA[Title/Abstract] OR shRNA[Title/Abstract])',
                f'({gene_name}[Title/Abstract]) AND ("small interfering RNA"[Title/Abstract] OR "short hairpin RNA"[Title/Abstract])',
                f'({gene_name}[Title/Abstract]) AND ("knockdown"[Title/Abstract] OR "silencing"[Title/Abstract]) AND RNAi',
                f'({gene_name}[Title/Abstract]) AND ("RNA interference"[Title/Abstract] OR "gene silencing"[Title/Abstract])'
            ]
            
            all_sequences = []
            seen_sequences = set()
            supplementary_only = []
            
            for term in search_terms:
                try:
                    # 搜索PubMed
                    search_result = ncbi_request('esearch.fcgi', {
                        'db': 'pubmed',
                        'term': term,
                        'retmax': 15,
                        'sort': 'relevance',
                        'retmode': 'json'
                    })
                    if not search_result:
                        continue
                    
                    pmid_list = search_result.get('esearchresult', {}).get('idlist', [])
                    if not pmid_list:
                        continue
                    
                    # 获取PMC ID
                    link_result = ncbi_request('elink.fcgi', {
                        'dbfrom': 'pubmed',
                        'db': 'pmc',
                        'id': pmid_list,
                        'retmode': 'json'
                    })
                    if not link_result:
                        continue
                    
                    # 解析PMC ID
                    pmc_ids = []
                    pmid_to_pmc = {}
                    for linkset in link_result.get('linksets', []):
                        pmid = linkset.get('ids', [''])[0]
                        for link in linkset.get('linksetdbs', []):
                            if link.get('linkname') == 'pubmed_pmc':
                                for pmc_link in link.get('links', []):
                                    pmc_id = str(pmc_link)
                                    if pmc_id:
                                        pmc_ids.append(pmc_id)
                                        pmid_to_pmc[pmid] = pmc_id
                    
                    if not pmc_ids:
                        continue
                    
                    # 获取文献基本信息
                    medline_text = ncbi_request('efetch.fcgi', {
                        'db': 'pubmed',
                        'id': pmid_list,
                        'rettype': 'medline',
                        'retmode': 'text'
                    })
                    
                    # 解析PubMed信息
                    pmid_info = {}
                    if medline_text:
                        for article in medline_text.split('\n\n'):
                            pmid_match = re.search(r'PMID- (\d+)', article)
                            if pmid_match:
                                pmid = pmid_match.group(1)
                                title_match = re.search(r'TI  - (.+?)(?:\n[A-Z]{2}  -|\Z)', article, re.DOTALL)
                                year_match = re.search(r'DP  - (\d{4})', article)
                                pmid_info[pmid] = {
                                    'title': title_match.group(1).replace('\n      ', ' ') if title_match else '',
                                    'year': year_match.group(1) if year_match else ''
                                }
                    
                    # 获取PMC全文
                    for pmid, pmc_id in pmid_to_pmc.items():
                        try:
                            pmc_xml = ncbi_request('efetch.fcgi', {
                                'db': 'pmc',
                                'id': pmc_id,
                                'rettype': 'xml',
                                'retmode': 'xml'
                            })
                            if not pmc_xml:
                                continue
                            
                            info = pmid_info.get(pmid, {'title': '', 'year': ''})
                            
                            # 查找Methods部分（使用正则）
                            methods_patterns = [
                                r'<sec[^>]*sec-type=["\']methods["\'][^>]*>(.*?)</sec>',
                                r'<title>\s*Materials?\s+and\s+Methods\s*</title>(.*?)(?:<sec>|<title>|<back>|</body>)',
                                r'<title>\s*Methods\s*</title>(.*?)(?:<sec>|<title>|<back>|</body>)',
                                r'<title>\s*Experimental\s+Procedures?\s*</title>(.*?)(?:<sec>|<title>|<back>|</body>)',
                            ]
                            
                            methods_text = ''
                            for pattern in methods_patterns:
                                matches = re.findall(pattern, pmc_xml, re.IGNORECASE | re.DOTALL)
                                if matches:
                                    methods_text = ' '.join(matches)
                                    break
                            
                            # 清理文本
                            methods_text = re.sub(r'<[^>]+>', ' ', methods_text)
                            methods_text = re.sub(r'\s+', ' ', methods_text)
                            
                            if not methods_text.strip():
                                continue
                            
                            # 检查是否在补充材料中
                            supplementary_patterns = [
                                r'(?:see|available|provided|listed|described)\s+(?:in|on)\s+(?:the\s+)?(?:Supplementary|supplemental|supporting|additional)',
                                r'Supplementary\s+(?:Table|Data|File|Material|Information)',
                                r'supplementary\s+data',
                                r'Table\s+S\d+',
                            ]
                            
                            has_supplementary = any(
                                re.search(p, methods_text, re.IGNORECASE) for p in supplementary_patterns
                            )
                            
                            # 搜索siRNA/shRNA序列 - 多种模式匹配
                            sequence_patterns = [
                                # siRNA序列模式 (19-25nt)
                                r'siRNA[s]?\s*[:=]\s*["\']?([ACGU]{19,25})["\']?',
                                r'siRNA[s]?\s+target[s]?\s*[:=]\s*["\']?([ACGU]{19,25})["\']?',
                                r'siRNA[s]?\s+sequence[s]?\s*[:=]\s*["\']?([ACGU]{19,25})["\']?',
                                # shRNA序列模式
                                r'shRNA[s]?\s*[:=]\s*["\']?([ACGU]{19,25})["\']?',
                                r'shRNA[s]?\s+target[s]?\s*[:=]\s*["\']?([ACGU]{19,25})["\']?',
                                r'hairpin\s+sequence[s]?\s*[:=]\s*["\']?([ACGU]{19,25})["\']?',
                                # 正义链/反义链模式
                                r'sense\s*[:=]\s*["\']?([ACGU]{19,25})["\']?',
                                r'antisense\s*[:=]\s*["\']?([ACGU]{19,25})["\']?',
                                # 通用靶序列模式
                                r'target\s+sequence[s]?\s*[:=]\s*["\']?([ACGU]{19,25})["\']?',
                                # 带位置信息的模式
                                rf'{gene_name}\s+[A-Za-z]*\s*[:=]?\s*["\']?([ACGU]{{19,25}})["\']?',
                                # 表格中的序列模式
                                r'[\(\[]?\d+[\)\[]?\s*[:=)\]]\s*["\']?([ACGU]{19,25})["\']?',
                            ]
                            
                            found_in_methods = False
                            for pattern in sequence_patterns:
                                matches = re.findall(pattern, methods_text, re.IGNORECASE)
                                for seq in matches:
                                    seq_upper = seq.upper()
                                    # 过滤条件
                                    if seq_upper in seen_sequences:
                                        continue
                                    if len(set(seq_upper)) < 4:  # 序列多样性检查
                                        continue
                                    # 过滤引物序列、接头序列等
                                    if seq_upper.startswith('TAATACGACTCACTATA') or 'AAAAAAAA' in seq_upper or 'TTTTTTTT' in seq_upper:
                                        continue
                                    # 确保是RNA序列（含U）或DNA序列（含T但转为RNA）
                                    if 'U' in seq_upper:
                                        rna_seq = seq_upper
                                    elif 'T' in seq_upper:
                                        rna_seq = seq_upper.replace('T', 'U')
                                    else:
                                        rna_seq = seq_upper
                                    
                                    seen_sequences.add(seq_upper)
                                    found_in_methods = True
                                    
                                    all_sequences.append({
                                        'target_seq': rna_seq,
                                        'target_region': '文献Methods部分报道',
                                        'design_rationale': '文献报道序列',
                                        'efficiency_score': '文献报道',
                                        'type': 'siRNA' if 'siRNA' in methods_text.upper() else 'shRNA',
                                        'source': 'literature',
                                        'location': 'Materials and Methods',
                                        'reference': {
                                            'type': '文献',
                                            'title': info['title'][:150] + '...' if len(info['title']) > 150 else info['title'],
                                            'year': info['year'],
                                            'pmid_or_patent': f'PMID:{pmid}',
                                            'url': f'https://pubmed.ncbi.nlm.nih.gov/{pmid}/',
                                            'pmc_url': f'https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}/'
                                        }
                                    })
                            
                            if not found_in_methods and has_supplementary and ('siRNA' in methods_text.upper() or 'shRNA' in methods_text.upper()):
                                supplementary_only.append({
                                    'title': info['title'][:150] + '...' if len(info['title']) > 150 else info['title'],
                                    'year': info['year'],
                                    'pmid': pmid,
                                    'url': f'https://pubmed.ncbi.nlm.nih.gov/{pmid}/',
                                    'pmc_url': f'https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}/',
                                    'note': 'siRNA/shRNA序列在补充材料中'
                                })
                            
                        except Exception as e:
                            continue
                                
                except Exception as e:
                    continue
            
            # ===== 搜索专利数据库 =====
            patent_search_result = self._search_patents_for_rnai(gene_name, seen_sequences)
            patent_sequences = patent_search_result.get('sequences', [])
            patent_metadata = patent_search_result.get('metadata', [])
            all_sequences.extend(patent_sequences)
            
            # 构建返回结果
            result = {
                'sequences': all_sequences[:5],
                'supplementary_only': supplementary_only[:3],
                'patent_metadata': patent_metadata[:5],
                'source': 'literature_and_patents',
                'search_location': 'PMC全文Materials and Methods + 专利权利要求书/详细说明',
                'shrna_vector_design': {
                    'loop_sequence': 'TTCAAGAGA（文献常用）',
                    'promoter': 'U6或H1（文献常用）',
                    'cloning_sites': 'BamHI/EcoRI 或 BbsI/BsmBI（文献常用）',
                    'notes': '建议使用pLKO.1或pSUPER载体系统（经典文献方案）'
                },
                'validation_method': 'Western blot或qPCR检测mRNA水平；建议通过原文确认序列准确性'
            }
            
            # 统计来源
            literature_count = len([s for s in all_sequences if s.get('source') == 'literature'])
            patent_count = len([s for s in all_sequences if s.get('source') == 'patent'])
            
            if all_sequences:
                result['notes'] = f'从{literature_count}篇文献和{patent_count}件专利中找到siRNA/shRNA序列。建议通过原文确认准确性。'
            elif patent_metadata:
                result['notes'] = f'发现{len(patent_metadata)}件相关专利可能包含siRNA/shRNA序列（需查阅权利要求书）。'
            elif supplementary_only:
                result['notes'] = f'发现{len(supplementary_only)}篇文献在Methods中提及siRNA/shRNA，但序列在补充材料中。'
            else:
                result['notes'] = '未在文献和专利全文中找到siRNA/shRNA序列。建议：1) 查阅文献补充材料；2) 使用在线工具设计。'
                result['alternative_tools'] = [
                    {'name': 'siDirect', 'url': 'https://sidirect2.rnai.jp/'},
                    {'name': 'BLOCK-iT RNAi Designer', 'url': 'https://www.thermofisher.com/cn/zh/home/life-science/rnai/rnai-design-tools.html'},
                    {'name': 'DSIR', 'url': 'http://biodev.extra.cea.fr/DSIR/DSIR.html'}
                ]
            
            return result
                
        except Exception as e:
            import traceback
            return {'error': str(e), 'sequences': [], 'source': 'literature', 'note': f'检索失败: {str(e)}', 'traceback': traceback.format_exc()[:500]}

    def _search_patents_for_rnai(self, gene_name: str, seen_sequences: set) -> Dict:
        """从专利数据库中搜索siRNA/shRNA序列"""
        patent_sequences = []
        patent_metadata = []
        
        try:
            try:
                from Bio import Entrez
            except ImportError:
                return {'sequences': [], 'metadata': []}
            
            import re
            
            patent_terms = [
                f'({gene_name}[Title]) AND (siRNA OR shRNA OR "RNA interference")',
                f'({gene_name}[Title]) AND ("small interfering RNA" OR "short hairpin RNA")',
                f'({gene_name}[Abstract]) AND (siRNA OR shRNA OR "gene silencing")',
            ]
            
            for term in patent_terms:
                try:
                    handle = Entrez.esearch(db='pat', term=term, retmax=10, sort='relevance')
                    record = Entrez.read(handle)
                    handle.close()
                    
                    patent_ids = record.get('IdList', [])
                    if not patent_ids:
                        continue
                    
                    # 获取专利详情
                    handle = Entrez.efetch(db='pat', id=patent_ids, rettype='xml', retmode='xml')
                    patent_xml = handle.read()
                    handle.close()
                    
                    if isinstance(patent_xml, bytes):
                        patent_xml = patent_xml.decode('utf-8')
                    
                    # 解析专利XML
                    patent_items = re.findall(r'<Patent[^>]*>(.*?)</Patent>', patent_xml, re.DOTALL)
                    
                    for item in patent_items:
                        # 提取专利标题
                        title_match = re.search(r'<PatentTitle[^>]*>(.*?)</PatentTitle>', item, re.DOTALL)
                        title = re.sub(r'<[^>]+>', '', title_match.group(1)) if title_match else ''
                        
                        # 提取专利号
                        patent_id_match = re.search(r'<PatentId[^>]*>(.*?)</PatentId>', item, re.DOTALL)
                        patent_id = patent_id_match.group(1) if patent_id_match else ''
                        
                        # 提取申请日期
                        date_match = re.search(r'<Date>(\d{4})', item)
                        year = date_match.group(1) if date_match else ''
                        
                        # 尝试提取权利要求书和详细说明
                        claims_match = re.search(r'<Claims[^>]*>(.*?)</Claims>', item, re.DOTALL)
                        description_match = re.search(r'<Description[^>]*>(.*?)</Description>', item, re.DOTALL)
                        abstract_match = re.search(r'<Abstract[^>]*>(.*?)</Abstract>', item, re.DOTALL)
                        
                        # 合并所有文本
                        full_text = ''
                        if claims_match:
                            claims_text = re.sub(r'<[^>]+>', ' ', claims_match.group(1))
                            full_text += claims_text + ' '
                        if description_match:
                            desc_text = re.sub(r'<[^>]+>', ' ', description_match.group(1))
                            full_text += desc_text + ' '
                        if abstract_match:
                            abs_text = re.sub(r'<[^>]+>', ' ', abstract_match.group(1))
                            full_text += abs_text + ' '
                        
                        # 如果没有找到详细文本，只记录元数据
                        if not full_text.strip():
                            check_text = (title + ' ' + (abstract_match.group(0) if abstract_match else '')).upper()
                            if any(kw in check_text for kw in ['SIRNA', 'SHRNA', 'RNAI', 'SILENCING', gene_name.upper()]):
                                patent_metadata.append({
                                    'type': 'patent_fulltext_needed',
                                    'title': title[:200] + '...' if len(title) > 200 else title,
                                    'year': year,
                                    'patent_id': patent_id,
                                    'note': '专利中可能包含siRNA/shRNA序列，需查阅全文（权利要求书/详细说明）',
                                    'url': f'https://patents.google.com/?q={patent_id.replace(" ", "+")}'
                                })
                            continue
                        
                        # 清理文本
                        full_text = re.sub(r'\s+', ' ', full_text)
                        
                        # 搜索siRNA/shRNA序列
                        sequence_patterns = [
                            r'siRNA[s]?\s*[:=]\s*["\']?([ACGU]{19,25})["\']?',
                            r'shRNA[s]?\s*[:=]\s*["\']?([ACGU]{19,25})["\']?',
                            r'target\s+sequence[s]?\s*[:=]\s*["\']?([ACGU]{19,25})["\']?',
                            r'["\']([ACGU]{19,25})["\'][^ACGU]*(?:siRNA|shRNA)',
                            r'(?:siRNA|shRNA)[^ACGU]*["\']([ACGU]{19,25})["\']',
                            r'claim\s+\d+[:\s]+[\s\S]{0,100}([ACGU]{19,25})',
                            r'sequence\s*(?:ID)?\s*(?:NO)?[:\s\.]*\d*[\s:]*([ACGU]{19,25})',
                        ]
                        
                        found_in_patent = False
                        for pattern in sequence_patterns:
                            matches = re.findall(pattern, full_text, re.IGNORECASE)
                            for seq in matches:
                                seq_upper = seq.upper()
                                
                                if seq_upper in seen_sequences:
                                    continue
                                if len(set(seq_upper)) < 4:
                                    continue
                                if seq_upper.startswith('TAATACGACTCACTATA'):
                                    continue
                                if 'AAAAAAAA' in seq_upper or 'TTTTTTTT' in seq_upper:
                                    continue
                                
                                # 确保是RNA序列
                                if 'U' in seq_upper:
                                    rna_seq = seq_upper
                                elif 'T' in seq_upper:
                                    rna_seq = seq_upper.replace('T', 'U')
                                else:
                                    rna_seq = seq_upper
                                    
                                seen_sequences.add(seq_upper)
                                found_in_patent = True
                                
                                patent_sequences.append({
                                    'target_seq': rna_seq,
                                    'target_region': '专利权利要求书/详细说明中报道',
                                    'design_rationale': '专利报道序列',
                                    'efficiency_score': '专利报道',
                                    'type': 'siRNA/shRNA',
                                    'source': 'patent',
                                    'location': '专利全文',
                                    'reference': {
                                        'type': '专利',
                                        'title': title[:200] + '...' if len(title) > 200 else title,
                                        'year': year,
                                        'pmid_or_patent': patent_id,
                                        'url': f'https://patents.google.com/?q={patent_id.replace(" ", "+")}'
                                    }
                                })
                        
                        # 如果文本中有siRNA/shRNA关键词但没找到序列，记录元数据
                        if not found_in_patent and ('SIRNA' in full_text.upper() or 'SHRNA' in full_text.upper()):
                            patent_metadata.append({
                                'type': 'patent_fulltext_needed',
                                'title': title[:200] + '...' if len(title) > 200 else title,
                                'year': year,
                                'patent_id': patent_id,
                                'note': '专利涉及siRNA/shRNA，但需查阅完整权利要求书获取序列',
                                'url': f'https://patents.google.com/?q={patent_id.replace(" ", "+")}'
                            })
                                
                except Exception as e:
                    continue
                    
        except Exception as e:
            pass
        
        # 合并结果：优先返回找到的序列，如果没有则返回元数据
        if patent_sequences:
            return {'sequences': patent_sequences, 'metadata': []}
        else:
            return {'sequences': [], 'metadata': patent_metadata[:5]}

    def design_crispr_sequences(self, gene_name: str, gene_id: str = "",
                               gene_description: str = "") -> Dict:
        """从PMC全文文献中检索Materials and Methods部分的sgRNA序列（严格仅使用文献报道的序列，禁止AI生成或推测）"""
        try:
            import re
            import xml.etree.ElementTree as ET
            
            # 设置NCBI API参数
            email, api_key, error = APIConfig.get_ncbi_credentials()
            if error:
                return {'error': 'NCBI API未配置', 'sgrnas': [], 'source': 'literature'}
            
            base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
            
            def ncbi_request(endpoint, params):
                """发送NCBI API请求"""
                params['tool'] = 'LentivirusAssessment'
                params['email'] = email
                if api_key:
                    params['api_key'] = api_key
                try:
                    response = requests.get(f"{base_url}/{endpoint}", params=params, timeout=30)
                    response.raise_for_status()
                    if params.get('retmode') == 'json':
                        return response.json()
                    return response.text
                except Exception as e:
                    logger.error(f"NCBI请求失败: {e}")
                    return None
            
            # 搜索PMC（PubMed Central）全文数据库
            search_terms = [
                f'({gene_name}[Title/Abstract]) AND (CRISPR[Title/Abstract] OR sgRNA[Title/Abstract])',
                f'({gene_name}[Title/Abstract]) AND ("single guide RNA"[Title/Abstract] OR "guide RNA"[Title/Abstract])',
                f'({gene_name}[Title/Abstract]) AND ("knockout"[Title/Abstract] OR "knock-out"[Title/Abstract]) AND CRISPR'
            ]
            
            all_sgrnas = []
            seen_sequences = set()
            supplementary_only = []
            
            for term in search_terms:
                try:
                    # 搜索PubMed
                    search_result = ncbi_request('esearch.fcgi', {
                        'db': 'pubmed',
                        'term': term,
                        'retmax': 15,
                        'sort': 'relevance',
                        'retmode': 'json'
                    })
                    if not search_result:
                        continue
                    
                    pmid_list = search_result.get('esearchresult', {}).get('idlist', [])
                    if not pmid_list:
                        continue
                    
                    # 获取PMC ID
                    link_result = ncbi_request('elink.fcgi', {
                        'dbfrom': 'pubmed',
                        'db': 'pmc',
                        'id': pmid_list,
                        'retmode': 'json'
                    })
                    if not link_result:
                        continue
                    
                    # 解析PMC ID
                    pmc_ids = []
                    pmid_to_pmc = {}
                    for linkset in link_result.get('linksets', []):
                        pmid = linkset.get('ids', [''])[0]
                        for link in linkset.get('linksetdbs', []):
                            if link.get('linkname') == 'pubmed_pmc':
                                for pmc_link in link.get('links', []):
                                    pmc_id = str(pmc_link)
                                    if pmc_id:
                                        pmc_ids.append(pmc_id)
                                        pmid_to_pmc[pmid] = pmc_id
                    
                    if not pmc_ids:
                        continue
                    
                    # 获取文献基本信息
                    medline_text = ncbi_request('efetch.fcgi', {
                        'db': 'pubmed',
                        'id': pmid_list,
                        'rettype': 'medline',
                        'retmode': 'text'
                    })
                    
                    # 解析PubMed信息
                    pmid_info = {}
                    if medline_text:
                        for article in medline_text.split('\n\n'):
                            pmid_match = re.search(r'PMID- (\d+)', article)
                            if pmid_match:
                                pmid = pmid_match.group(1)
                                title_match = re.search(r'TI  - (.+?)(?:\n[A-Z]{2}  -|\Z)', article, re.DOTALL)
                                year_match = re.search(r'DP  - (\d{4})', article)
                                pmid_info[pmid] = {
                                    'title': title_match.group(1).replace('\n      ', ' ') if title_match else '',
                                    'year': year_match.group(1) if year_match else ''
                                }
                    
                    # 获取PMC全文
                    for pmid, pmc_id in pmid_to_pmc.items():
                        try:
                            pmc_xml = ncbi_request('efetch.fcgi', {
                                'db': 'pmc',
                                'id': pmc_id,
                                'rettype': 'xml',
                                'retmode': 'xml'
                            })
                            if not pmc_xml:
                                continue
                            
                            info = pmid_info.get(pmid, {'title': '', 'year': ''})
                            
                            # 查找Methods部分（使用正则）
                            methods_patterns = [
                                r'<sec[^>]*sec-type=["\']methods["\'][^>]*>(.*?)</sec>',
                                r'<title>\s*Materials?\s+and\s+Methods\s*</title>(.*?)(?:<sec>|<title>|<back>|</body>)',
                                r'<title>\s*Methods\s*</title>(.*?)(?:<sec>|<title>|<back>|</body>)',
                                r'<title>\s*Experimental\s+Procedures?\s*</title>(.*?)(?:<sec>|<title>|<back>|</body>)',
                            ]
                            
                            methods_text = ''
                            for pattern in methods_patterns:
                                matches = re.findall(pattern, pmc_xml, re.IGNORECASE | re.DOTALL)
                                if matches:
                                    methods_text = ' '.join(matches)
                                    break
                            
                            # 清理文本
                            methods_text = re.sub(r'<[^>]+>', ' ', methods_text)
                            methods_text = re.sub(r'\s+', ' ', methods_text)
                            
                            if not methods_text.strip():
                                continue
                            
                            # 检查是否在补充材料中
                            supplementary_patterns = [
                                r'(?:see|available|provided|listed|described)\s+(?:in|on)\s+(?:the\s+)?(?:Supplementary|supplemental|supporting|additional)',
                                r'Supplementary\s+(?:Table|Data|File|Material|Information)',
                                r'supplementary\s+data',
                                r'Table\s+S\d+',
                            ]
                            
                            has_supplementary = any(
                                re.search(p, methods_text, re.IGNORECASE) for p in supplementary_patterns
                            )
                            
                            # 搜索sgRNA序列
                            sequence_patterns = [
                                r'sgRNA[s]?\s*[:=]\s*["\']?([ACGT]{20})["\']?',
                                r'guide\s+RNA[s]?\s*[:=]\s*["\']?([ACGT]{20})["\']?',
                                r'target\s+sequence[s]?\s*[:=]\s*["\']?([ACGT]{20})["\']?',
                                r'["\']([ACGT]{20})["\'][^ACGT]*(?:sgRNA|guide|target)',
                                r'(?:sgRNA|guide|target)[^ACGT]*["\']([ACGT]{20})["\']',
                                r'([ACGT]{20})\s+NGG',
                                rf'{gene_name}\s+[:\s]\s*([ACGT]{{20}})',
                                r'(?:sgRNA[_-]?)?\d+[:\s)]+([ACGT]{20})',
                            ]
                            
                            found_in_methods = False
                            for pattern in sequence_patterns:
                                matches = re.findall(pattern, methods_text, re.IGNORECASE)
                                for seq in matches:
                                    seq_upper = seq.upper()
                                    if seq_upper in seen_sequences:
                                        continue
                                    if len(set(seq_upper)) < 4:
                                        continue
                                    if seq_upper.startswith('TAATACGACTCACTATA'):
                                        continue
                                    if 'AAAAAAAA' in seq_upper or 'TTTTTTTT' in seq_upper:
                                        continue
                                    
                                    seen_sequences.add(seq_upper)
                                    found_in_methods = True
                                    
                                    all_sgrnas.append({
                                        'sequence': seq_upper,
                                        'pam': 'NGG (SpCas9)',
                                        'target_exon': '文献Methods部分报道',
                                        'efficiency_score': '文献报道',
                                        'off_target_risk': '未知',
                                        'source': 'literature',
                                        'location': 'Materials and Methods',
                                        'reference': {
                                            'type': '文献',
                                            'title': info['title'][:150] + '...' if len(info['title']) > 150 else info['title'],
                                            'year': info['year'],
                                            'pmid_or_patent': f'PMID:{pmid}',
                                            'url': f'https://pubmed.ncbi.nlm.nih.gov/{pmid}/',
                                            'pmc_url': f'https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}/'
                                        }
                                    })
                            
                            if not found_in_methods and has_supplementary and 'sgRNA' in methods_text.upper():
                                supplementary_only.append({
                                    'title': info['title'][:150] + '...' if len(info['title']) > 150 else info['title'],
                                    'year': info['year'],
                                    'pmid': pmid,
                                    'url': f'https://pubmed.ncbi.nlm.nih.gov/{pmid}/',
                                    'pmc_url': f'https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}/',
                                    'note': 'sgRNA序列在补充材料中'
                                })
                            
                        except Exception as e:
                            continue
                                
                except Exception as e:
                    continue
            
            # ===== 搜索专利数据库 =====
            patent_search_result = self._search_patents_for_sgrna(gene_name, seen_sequences)
            patent_sgrnas = patent_search_result.get('sequences', [])
            patent_metadata = patent_search_result.get('metadata', [])
            all_sgrnas.extend(patent_sgrnas)
            
            # 构建返回结果
            result = {
                'sgrnas': all_sgrnas[:5],
                'supplementary_only': supplementary_only[:3],
                'patent_metadata': patent_metadata[:5],
                'source': 'literature_and_patents',
                'search_location': 'PMC全文Materials and Methods + 专利权利要求书/详细说明',
                'lentivirus_vector': {
                    'backbone': 'lentiCRISPRv2或pLentiGuide-Puro',
                    'promoter': 'U6',
                    'selection': 'Puromycin',
                    'cloning_strategy': 'BsmBI酶切（粘性末端：CACC/GTGT）'
                },
                'validation_method': '1) T7E1酶切或Sanger测序验证切割效率；2) qPCR检测目标基因表达；3) Western Blot检测蛋白水平'
            }
            
            # 统计来源
            literature_count = len([s for s in all_sgrnas if s.get('source') == 'literature'])
            patent_count = len([s for s in all_sgrnas if s.get('source') == 'patent'])
            
            if all_sgrnas:
                result['notes'] = f'从{literature_count}篇文献和{patent_count}件专利中找到sgRNA序列。建议通过原文确认准确性。'
            elif patent_metadata:
                result['notes'] = f'发现{len(patent_metadata)}件相关专利可能包含sgRNA序列（需查阅权利要求书）。'
            elif supplementary_only:
                result['notes'] = f'发现{len(supplementary_only)}篇文献在Methods中提及sgRNA，但序列在补充材料中。'
            else:
                result['notes'] = '未在文献和专利全文中找到sgRNA序列。建议：1) 查阅文献补充材料；2) 使用在线工具设计。'
                result['alternative_tools'] = [
                    {'name': 'CRISPOR', 'url': 'http://crispor.tefor.net'},
                    {'name': 'Benchling', 'url': 'https://benchling.com/crispr'},
                    {'name': 'Broad Institute SGE', 'url': 'https://sanger.sgc.utoronto.ca/sge/'}
                ]
            
            return result
                
        except Exception as e:
            import traceback
            return {'error': str(e), 'sgrnas': [], 'source': 'literature', 'note': f'检索失败: {str(e)}', 'traceback': traceback.format_exc()[:500]}

    def _search_patents_for_sgrna(self, gene_name: str, seen_sequences: set) -> list:
        """从专利数据库中搜索sgRNA序列，尝试获取权利要求书和详细说明"""
        patent_sgrnas = []
        patent_metadata = []  # 用于存储无法获取全文的专利信息
        
        try:
            try:
                from Bio import Entrez
            except ImportError:
                return {'sequences': [], 'metadata': []}
            
            import re
            
            patent_terms = [
                f'({gene_name}[Title]) AND (CRISPR OR sgRNA)',
                f'({gene_name}[Title]) AND ("guide RNA" OR "single guide RNA")',
                f'({gene_name}[Abstract]) AND (CRISPR OR sgRNA)',
            ]
            
            for term in patent_terms:
                try:
                    handle = Entrez.esearch(db='pat', term=term, retmax=10, sort='relevance')
                    record = Entrez.read(handle)
                    handle.close()
                    
                    patent_ids = record.get('IdList', [])
                    if not patent_ids:
                        continue
                    
                    # 获取专利详情 - 尝试获取完整文本
                    handle = Entrez.efetch(db='pat', id=patent_ids, rettype='xml', retmode='xml')
                    patent_xml = handle.read()
                    handle.close()
                    
                    if isinstance(patent_xml, bytes):
                        patent_xml = patent_xml.decode('utf-8')
                    
                    # 解析专利XML - 尝试提取权利要求书和详细说明
                    patent_items = re.findall(r'<Patent[^>]*>(.*?)</Patent>', patent_xml, re.DOTALL)
                    
                    for item in patent_items:
                        # 提取专利标题
                        title_match = re.search(r'<PatentTitle[^>]*>(.*?)</PatentTitle>', item, re.DOTALL)
                        title = re.sub(r'<[^>]+>', '', title_match.group(1)) if title_match else ''
                        
                        # 提取专利号
                        patent_id_match = re.search(r'<PatentId[^>]*>(.*?)</PatentId>', item, re.DOTALL)
                        patent_id = patent_id_match.group(1) if patent_id_match else ''
                        
                        # 提取申请日期
                        date_match = re.search(r'<Date>(\d{4})', item)
                        year = date_match.group(1) if date_match else ''
                        
                        # 尝试提取权利要求书和详细说明
                        claims_match = re.search(r'<Claims[^>]*>(.*?)</Claims>', item, re.DOTALL)
                        description_match = re.search(r'<Description[^>]*>(.*?)</Description>', item, re.DOTALL)
                        abstract_match = re.search(r'<Abstract[^>]*>(.*?)</Abstract>', item, re.DOTALL)
                        
                        # 合并所有文本
                        full_text = ''
                        if claims_match:
                            claims_text = re.sub(r'<[^>]+>', ' ', claims_match.group(1))
                            full_text += claims_text + ' '
                        if description_match:
                            desc_text = re.sub(r'<[^>]+>', ' ', description_match.group(1))
                            full_text += desc_text + ' '
                        if abstract_match:
                            abs_text = re.sub(r'<[^>]+>', ' ', abstract_match.group(1))
                            full_text += abs_text + ' '
                        
                        # 如果没有找到详细文本，只记录元数据
                        if not full_text.strip():
                            check_text = (title + ' ' + (abstract_match.group(0) if abstract_match else '')).upper()
                            if any(kw in check_text for kw in ['SGRNA', 'GUIDE RNA', 'CRISPR', gene_name.upper()]):
                                patent_metadata.append({
                                    'type': 'patent_fulltext_needed',
                                    'title': title[:200] + '...' if len(title) > 200 else title,
                                    'year': year,
                                    'patent_id': patent_id,
                                    'note': '专利中可能包含sgRNA序列，需查阅全文（权利要求书/详细说明）',
                                    'url': f'https://patents.google.com/?q={patent_id.replace(" ", "+")}'
                                })
                            continue
                        
                        # 清理文本
                        full_text = re.sub(r'\s+', ' ', full_text)
                        
                        # 搜索sgRNA序列
                        sequence_patterns = [
                            r'sgRNA[s]?\s*[:=]\s*["\']?([ACGT]{20})["\']?',
                            r'guide\s+RNA[s]?\s*[:=]\s*["\']?([ACGT]{20})["\']?',
                            r'target\s+sequence[s]?\s*[:=]\s*["\']?([ACGT]{20})["\']?',
                            r'["\']([ACGT]{20})["\'][^ACGT]*(?:sgRNA|guide)',
                            r'([ACGT]{20})\s+NGG',
                            r'claim\s+\d+[:\s]+[\s\S]{0,100}([ACGT]{20})',
                            r'sequence\s*(?:ID)?\s*(?:NO)?[:\s\.]*\d*[\s:]*([ACGT]{20})',
                        ]
                        
                        found_in_patent = False
                        for pattern in sequence_patterns:
                            matches = re.findall(pattern, full_text, re.IGNORECASE)
                            for seq in matches:
                                seq_upper = seq.upper()
                                
                                if seq_upper in seen_sequences:
                                    continue
                                if len(set(seq_upper)) < 4:
                                    continue
                                if seq_upper.startswith('TAATACGACTCACTATA'):
                                    continue
                                if 'AAAAAAAA' in seq_upper or 'TTTTTTTT' in seq_upper:
                                    continue
                                    
                                seen_sequences.add(seq_upper)
                                found_in_patent = True
                                
                                patent_sgrnas.append({
                                    'sequence': seq_upper,
                                    'pam': 'NGG (SpCas9)',
                                    'target_exon': '专利权利要求书/详细说明中报道',
                                    'efficiency_score': '专利报道',
                                    'off_target_risk': '未知',
                                    'source': 'patent',
                                    'location': '专利全文',
                                    'reference': {
                                        'type': '专利',
                                        'title': title[:200] + '...' if len(title) > 200 else title,
                                        'year': year,
                                        'pmid_or_patent': patent_id,
                                        'url': f'https://patents.google.com/?q={patent_id.replace(" ", "+")}'
                                    }
                                })
                        
                        # 如果文本中有sgRNA关键词但没找到序列，记录元数据
                        if not found_in_patent and 'SGRNA' in full_text.upper():
                            patent_metadata.append({
                                'type': 'patent_fulltext_needed',
                                'title': title[:200] + '...' if len(title) > 200 else title,
                                'year': year,
                                'patent_id': patent_id,
                                'note': '专利涉及sgRNA，但需查阅完整权利要求书获取序列',
                                'url': f'https://patents.google.com/?q={patent_id.replace(" ", "+")}'
                            })
                                
                except Exception as e:
                    continue
                    
        except Exception as e:
            pass
        
        # 合并结果：优先返回找到的序列，如果没有则返回元数据
        if patent_sgrnas:
            return {'sequences': patent_sgrnas, 'metadata': []}
        else:
            return {'sequences': [], 'metadata': patent_metadata[:5]}

    def analyze_cell_culture_difficulty(self, cell_line: str, papers: List[Dict]) -> Dict:
        if not self.api_key:
            return {'error': '未配置AI API', 'note': '无法分析培养难点'}

        literature_text = ""
        if papers:
            literature_text = "\n".join([
                f"文献{i+1}: {p.get('title', '')}\n摘要: {p.get('abstract', '')[:400]}..."
                for i, p in enumerate(papers[:5])
            ])

        checklist_text = ""
        for category, items in self.CELL_CULTURE_DIFFICULTY_CHECKLIST.items():
            checklist_text += f"\n{category}:\n" + "\n".join([f"- {item}" for item in items])

        # 【严格模式】如果没有文献，返回空结果
        if not literature_text:
            return {
                'culture_medium': [],
                'coating_matrix': [],
                'environment': [],
                'operation': [],
                'time_cost': [],
                'special_warnings': ['未检索到该细胞系的培养文献，无法提供可靠信息'],
                'protocol_tips': ['建议查阅ATCC官方资料或相关protocol文献'],
                'verified_by': [],
                'data_source_note': '未找到相关文献，AI未进行推测'
            }

        prompt = f"""作为细胞培养专家，请【严格基于以下提供的文献】分析细胞系"{cell_line}"的培养难点，并进行语义分析和归纳总结。

【重要警告】你绝对不能进行推测或编造：
1. 如果文献中未提及某类信息，该字段必须返回空数组 []
2. 每个描述必须标注文献来源（PMID）
3. 禁止基于"常识"或"一般特点"进行推断

提供的文献：
{literature_text}

请按以下JSON格式返回（只返回JSON）：
{{
    "culture_medium": [
        "文献明确报道的难点 + [PMID:XXXX]"
    ],
    "coating_matrix": [
        "文献明确报道的难点 + [PMID:XXXX]"
    ],
    "environment": [
        "文献明确报道的难点 + [PMID:XXXX]"
    ],
    "operation": [
        "文献明确报道的难点 + [PMID:XXXX]"
    ],
    "time_cost": [
        "文献明确报道的难点 + [PMID:XXXX]"
    ],
    "special_warnings": [
        "文献明确报道的警告 + [PMID:XXXX]"
    ],
    "protocol_tips": [
        "文献明确报道的建议 + [PMID:XXXX]"
    ],
    "verified_by": [
        "PMID:XXXX（文献标题）"
    ],
    "data_source_note": "基于X篇文献分析，所有信息均有文献支持",
    
    "semantic_analysis": {{
        "difficulty_assessment": "基于文献内容的培养难度语义评估（Simple/Moderate/Complex/Highly_Demanding）",
        "key_constraints": ["文献中隐含的关键限制因素"],
        "critical_success_factors": ["文献强调的成功培养关键要素"],
        "risk_indicators": ["文献中提到的风险信号或预警信号"],
        "contextual_notes": "文献中描述的实验背景对培养的影响"
    }},
    
    "inductive_summary": {{
        "primary_findings": "文献中反复出现的主要发现",
        "consensus_points": "多篇文献的共识性观点",
        "conflicting_evidence": "文献间存在分歧的证据（如有）",
        "knowledge_gaps": "文献中未覆盖的信息缺口",
        "practical_recommendations": "基于文献整合的实际操作建议"
    }}
}}

严格要求：
1. 【禁止推测】没有文献支持的信息一律不输出，返回空数组
2. 【必须标注来源】每个具体描述后必须标注[PMID:XXXX]
3. 【禁止常识推断】即使是常见细胞系（如HK-2、HeLa），没有文献支持不得输出
4. 文献中未提及的类别，必须返回空数组 []
5. 【语义分析】基于文献内容进行深度语义理解，提取隐含信息和上下文含义
6. 【归纳总结】对多篇文献的信息进行整合归纳，找出模式、共识和差异"""

        try:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            payload = {
                'model': 'qwen-turbo',
                'input': {
                    'messages': [
                        {'role': 'system', 'content': '你是细胞培养技术专家，熟悉各种细胞系的培养难点和protocol，包括HK-2（人肾小管上皮细胞）、NCI-H226（肺鳞癌细胞）等常见细胞系的特殊要求。'},
                        {'role': 'user', 'content': prompt}
                    ]
                },
                'parameters': {
                    'result_format': 'message',
                    'max_tokens': 2000,
                    'temperature': 0.2
                }
            }

            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()

            result = response.json()
            content = result.get('output', {}).get('choices', [{}])[0].get('message', {}).get('content', '')

            try:
                content_clean = content.replace('```json', '').replace('```', '').strip()
                analysis = json.loads(content_clean)
                if 'data_source_note' not in analysis:
                    analysis['data_source_note'] = '基于AI分析' if not papers else '基于文献分析'
                return analysis
            except json.JSONDecodeError:
                return {'error': '格式解析失败', 'raw': content[:500], 'data_source_note': '解析失败'}

        except Exception as e:
            return {'error': str(e), 'data_source_note': 'API调用失败'}

    def analyze_lentivirus_susceptibility(self, cell_line: str, papers: List[Dict]) -> Dict:
        if not self.api_key:
            return {'error': '未配置AI API', 'note': '无法分析易感性'}

        literature_text = ""
        if papers:
            literature_text = "\n".join([
                f"文献{i+1}: {p.get('title', '')}\n摘要: {p.get('abstract', '')[:400]}..."
                for i, p in enumerate(papers[:5])
            ])

        # 【严格模式】如果没有文献，返回空结果
        if not literature_text:
            return {
                'susceptibility_level': 'Unknown',
                'recommended_moi': '未检索到文献，无法推荐',
                'infection_efficiency': '未检索到文献，无法评估',
                'requires_polybrene': '未知',
                'requires_spinfection': '未知',
                'requires_pseudotyping': '未知',
                'cell_line_info': cell_line,
                'challenges': ['未检索到该细胞系的慢病毒感染文献，无法提供可靠信息'],
                'optimization_tips': ['建议查阅文献获取该细胞系的慢病毒感染条件'],
                'reported_cell_lines': [],
                'references': [],
                'data_source_note': '未找到相关文献，AI未进行推测'
            }

        prompt = f"""分析细胞系"{cell_line}"的慢病毒/逆转录病毒易感性，并进行语义分析和归纳总结。

【重要警告】你绝对不能进行推测或编造：
1. 只能基于提供的文献回答问题
2. 如果文献未提及某项信息，必须标注"文献未提供"
3. 每个具体数据必须标注文献来源（PMID）

基于以下文献：
{literature_text}

请提供以下信息（JSON格式）：
{{
    "susceptibility_level": "High/Medium/Low/Unknown（基于文献判断，无文献则为Unknown）",
    "recommended_moi": "文献明确报道的MOI范围 + [PMID:XXXX]",
    "infection_efficiency": "文献明确报道的效率 + [PMID:XXXX]",
    "requires_polybrene": "文献明确报道（是/否/未知） + [PMID:XXXX]",
    "requires_spinfection": "文献明确报道（是/否/未知） + [PMID:XXXX]",
    "requires_pseudotyping": "文献明确报道的特殊包膜蛋白 + [PMID:XXXX]",
    "cell_line_info": "文献中明确描述的信息",
    "challenges": [
        "文献明确报道的难点 + [PMID:XXXX]"
    ],
    "optimization_tips": [
        "文献明确报道的优化方法 + [PMID:XXXX]"
    ],
    "reported_cell_lines": [
        "文献中提到的相关细胞系 + [PMID:XXXX]"
    ],
    "references": ["PMID:XXXX（文献标题）"],
    "data_source_note": "基于X篇文献分析，所有信息均有文献支持",
    
    "semantic_analysis": {{
        "infection_profile": "基于文献的感染特性语义描述（如：易感性高但效率不稳定）",
        "critical_parameters": ["文献强调的关键感染参数"],
        "failure_modes": ["文献中描述的感染失败模式"],
        "optimization_potential": "文献暗示的优化潜力评估"
    }},
    
    "inductive_summary": {{
        "successful_strategies": "文献中反复出现的成功感染策略",
        "common_pitfalls": "多篇文献共同指出的常见问题",
        "efficiency_patterns": "感染效率的文献模式总结",
        "experimental_context": "文献中实验条件对结果的影响"
    }}
}}

严格要求：
1. 【禁止推测】文献未明确报道的信息必须标注"文献未提供"或"Unknown"
2. 【必须标注来源】每个具体数据后必须标注[PMID:XXXX]
3. 【禁止常识推断】即使是一般规律，没有文献支持不得输出
4. 如果文献未提供足够信息，返回"Unknown"而非推测
5. 【语义分析】深度理解文献中的隐含信息和上下文
6. 【归纳总结】整合多篇文献的发现，提取共性规律和差异"""

        try:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            payload = {
                'model': 'qwen-turbo',
                'input': {
                    'messages': [
                        {'role': 'system', 'content': '你是病毒学专家，精通慢病毒载体和细胞转导，熟悉各种细胞系的慢病毒感染特性。'},
                        {'role': 'user', 'content': prompt}
                    ]
                },
                'parameters': {
                    'result_format': 'message',
                    'max_tokens': 1500,
                    'temperature': 0.2
                }
            }

            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()

            result = response.json()
            content = result.get('output', {}).get('choices', [{}])[0].get('message', {}).get('content', '')

            try:
                content_clean = content.replace('```json', '').replace('```', '').strip()
                return json.loads(content_clean)
            except json.JSONDecodeError:
                return {'error': '格式解析失败', 'raw': content[:500]}

        except Exception as e:
            return {'error': str(e)}

# ==================== 核心数据库（第一层） ====================
class CoreDatabases:
    """核心基因数据库 - 包含已发表的文献支持的基因分类
    
    注意：以下PMID基于已发表的科学文献，但用户应通过PubMed验证最新信息
    """
    CORE_ESSENTIAL = {
        'ACTB': ('PMID:30971823', 'DepMap核心必需', '细胞骨架结构蛋白'),
        'GAPDH': ('PMID:30971823', 'DepMap核心必需', '糖酵解关键酶'),
        'HSP90AA1': ('PMID:30971823', 'DepMap核心必需', '分子伴侣'),
        'RPL11': ('PMID:30971823', 'DepMap核心必需', '核糖体大亚基蛋白'),
        'RPS3': ('PMID:30971823', 'DepMap核心必需', '核糖体小亚基蛋白'),
        'PCNA': ('PMID:30971823', 'DepMap核心必需', 'DNA复制辅助蛋白'),
        'TOP2A': ('PMID:30971823', 'DepMap核心必需', 'DNA拓扑异构酶II'),
        'AURKB': ('PMID:30971823', 'DepMap核心必需', '有丝分裂激酶'),
        'PLK1': ('PMID:30971823', 'DepMap核心必需', '细胞周期调控激酶'),
        'BUB1': ('PMID:30971823', 'DepMap核心必需', '纺锤体检查点'),
        'CDC20': ('PMID:30971823', 'DepMap核心必需', '细胞周期后期促进复合物'),
        'CHEK1': ('PMID:30971823', 'DepMap核心必需', 'DNA损伤检查点'),
        'KIF11': ('PMID:30971823', 'DepMap核心必需', '有丝分裂驱动蛋白'),
        'PSMD1': ('PMID:30971823', 'DepMap核心必需', '蛋白酶体亚基'),
        'POLR2A': ('PMID:30971823', 'DepMap核心必需', 'RNA聚合酶II最大亚基'),
    }

    CORE_TOXIC = {
        'BAX': ('PMID:10625696', '促凋亡Bcl-2家族', '过表达直接激活线粒体凋亡途径'),
        'BAK1': ('PMID:10625696', '促凋亡Bcl-2家族', '线粒体外膜通透化诱导凋亡'),
        'BID': ('PMID:10625696', '促凋亡BH3-only蛋白', '连接死亡受体到线粒体凋亡'),
        'PUMA': ('PMID:12968034', 'p53下游促凋亡', '强力促凋亡BH3-only蛋白'),
        'NOXA': ('PMID:12968034', 'p53下游促凋亡', '促凋亡BH3-only蛋白'),
        'CASP3': ('PMID:9228057', '凋亡执行caspase', '过表达直接激活凋亡级联反应'),
        'CASP7': ('PMID:9228057', '凋亡执行caspase', '细胞凋亡执行分子'),
        'CASP8': ('PMID:9228057', '凋亡启动caspase', '死亡受体通路启动分子'),
        'CASP9': ('PMID:9228057', '凋亡启动caspase', '线粒体通路启动分子'),
        'FAS': ('PMID:8666142', '死亡受体', '激活外源性凋亡途径'),
        'TNF': ('PMID:15157675', '促炎细胞因子', '诱导细胞毒性凋亡'),
        'TRAIL': ('PMID:10578115', 'TNF家族凋亡诱导配体', '选择性诱导肿瘤细胞凋亡'),
        'TP53': ('PMID:20154749', '肿瘤抑制基因', '过表达诱导G1阻滞和凋亡'),
        'CDKN1A': ('PMID:8242752', '细胞周期抑制基因', 'p21强抑制剂导致细胞周期停滞'),
        'PARP1': ('PMID:16794554', 'DNA修复酶', '过度激活导致NAD+耗竭和细胞死亡'),
    }

    CORE_ANTIVIRAL = {
        'MX1': ('PMID:21694717', 'ISG-I型干扰素诱导蛋白', '抑制流感病毒等RNA病毒复制'),
        'MX2': ('PMID:21694717', 'ISG-I型干扰素诱导蛋白', '抑制HIV-1等逆转录病毒'),
        'OAS1': ('PMID:21694717', 'ISG-寡腺苷酸合成酶', '激活RNase L降解病毒RNA'),
        'OAS2': ('PMID:21694717', 'ISG-寡腺苷酸合成酶', '抗病毒先天免疫效应分子'),
        'OAS3': ('PMID:21694717', 'ISG-寡腺苷酸合成酶', '抗病毒先天免疫效应分子'),
        'RNASEL': ('PMID:21694717', 'ISG-核糖核酸酶L', 'OAS通路下游降解病毒RNA'),
        'ISG15': ('PMID:21694717', 'ISG-干扰素刺激基因15', 'ISG化修饰抑制病毒复制'),
        'IFIT1': ('PMID:21694717', 'ISG-干扰素诱导蛋白', '抑制病毒翻译起始'),
        'IFIT2': ('PMID:21694717', 'ISG-干扰素诱导蛋白', '抑制病毒蛋白合成'),
        'IFIH1': ('PMID:21694717', 'ISG-MDA5', '识别病毒dsRNA激活免疫反应'),
        'DDX58': ('PMID:21694717', 'ISG-RIG-I', '识别病毒RNA诱导I型干扰素'),
        'TRIM5': ('PMID:15890885', '限制因子', '抑制HIV-1等逆转录病毒复制'),
        'APOBEC3G': ('PMID:12134021', '限制因子', '胞嘧啶脱氨酶抑制HIV-1（Vif靶向）'),
        'BST2': ('PMID:19543227', '限制因子', 'Tetherin抑制病毒出芽'),
        'KLF5': ('PMID:33597534', '转录因子-间接抗病毒', '调控IFITM家族和脂质代谢抑制SARS-CoV-2等病毒复制'),
    }

    @classmethod
    def check_gene(cls, gene_name: str, check_type: str) -> Optional[Tuple[str, str, str]]:
        gene_upper = gene_name.upper()
        if check_type == 'essential' and gene_upper in cls.CORE_ESSENTIAL:
            return cls.CORE_ESSENTIAL[gene_upper]
        elif check_type == 'toxic' and gene_upper in cls.CORE_TOXIC:
            return cls.CORE_TOXIC[gene_upper]
        elif check_type == 'antiviral' and gene_upper in cls.CORE_ANTIVIRAL:
            return cls.CORE_ANTIVIRAL[gene_upper]
        return None

# ==================== 安全配置 ====================
class SecurityConfig:
    GENE_NAME_PATTERN = re.compile(r'^[a-zA-Z][a-zA-Z0-9]*(-?[a-zA-Z0-9]+)*$')
    MAX_GENE_LENGTH = 50

    @staticmethod
    def sanitize_input(text: str, max_length: int = 100) -> str:
        if not text:
            return ""
        text = text.strip()[:max_length]
        text = ''.join(char for char in text if ord(char) >= 32 and char not in ['<', '>', '"', "'"])
        return text

    @staticmethod
    def validate_gene_name(gene_name: str) -> Tuple[bool, str]:
        if not gene_name:
            return False, "基因名不能为空"
        if len(gene_name) > SecurityConfig.MAX_GENE_LENGTH:
            return False, f"基因名过长（最大{SecurityConfig.MAX_GENE_LENGTH}字符）"
        if not SecurityConfig.GENE_NAME_PATTERN.match(gene_name):
            return False, "基因名格式无效（必须以字母开头）"
        return True, ""

# ==================== 密码验证 ====================
class AuthManager:
    @staticmethod
    def check_password():
        def password_entered():
            try:
                correct_password = st.secrets.get("APP_PASSWORD", "lenti2024")
            except:
                correct_password = "lenti2024"

            if st.session_state.get("password") == correct_password:
                st.session_state["password_correct"] = True
                if "password" in st.session_state:
                    del st.session_state["password"]
            else:
                st.session_state["password_correct"] = False

        if "password_correct" not in st.session_state:
            st.text_input("请输入访问密码", type="password", on_change=password_entered, key="password")
            st.info("请输入密码以访问系统")
            return False
        elif not st.session_state["password_correct"]:
            st.text_input("请输入访问密码", type="password", on_change=password_entered, key="password")
            st.error("密码错误，请重试")
            return False
        else:
            return True

# ==================== HPA数据管理（修复版 - 移除rna_celline依赖） ====================
# ==================== HPA数据管理（简化版 - 仅用于下载和数据路径）====================
class HPADataManager:
    """HPA数据管理 - 简化版（仅保留下载功能）"""
    HPA_URL = "https://www.proteinatlas.org/download/proteinatlas.tsv.zip"
    LOCAL_DIR = "hpa_data"

    def __init__(self):
        self.local_dir = self.LOCAL_DIR
        self.data_file = os.path.join(self.local_dir, "proteinatlas.tsv")
        self._init_storage()

    def _init_storage(self):
        if not os.path.exists(self.local_dir):
            os.makedirs(self.local_dir)

    def check_and_download(self) -> Dict:
        """检查并下载HPA数据，返回状态信息"""
        status = {
            'exists': False,
            'downloaded': False,
            'error': None,
            'file_path': self.data_file
        }
        
        if os.path.exists(self.data_file):
            status['exists'] = True
            return status
        
        # 尝试下载
        download_result = self._download_hpa_data()
        status['downloaded'] = download_result.get('success', False)
        status['error'] = download_result.get('error')
        status['file_path'] = self.data_file  # 可能在下载过程中更新了路径
        status['exists'] = os.path.exists(self.data_file)
        
        return status

    def _download_hpa_data(self) -> Dict:
        """下载proteinatlas.tsv，返回下载结果状态"""
        result = {'success': False, 'error': None}
        try:
            logger.info("【HPA下载】开始下载HPA数据库（proteinatlas.tsv，约200MB）...")
            zip_path = os.path.join(self.local_dir, "proteinatlas.tsv.zip")

            response = requests.get(self.HPA_URL, stream=True, timeout=300)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0

            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0 and downloaded % (1024*1024) == 0:
                            progress = (downloaded / total_size) * 100
                            logger.info(f"【HPA下载】进度: {progress:.1f}%")

            logger.info("【HPA下载】正在解压HPA数据...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.local_dir)

            # 验证文件是否存在
            if not os.path.exists(self.data_file):
                extracted_files = os.listdir(self.local_dir)
                for f in extracted_files:
                    if f.endswith('.tsv') and 'proteinatlas' in f.lower():
                        self.data_file = os.path.join(self.local_dir, f)
                        logger.info(f"【HPA下载】使用实际文件名: {f}")
                        break

            os.remove(zip_path)
            
            # 验证最终文件
            if os.path.exists(self.data_file):
                file_size = os.path.getsize(self.data_file)
                logger.info(f"【HPA下载】✓ 下载完成，文件大小: {file_size / (1024*1024):.1f} MB")
                result['success'] = True
            else:
                result['error'] = '解压后未找到proteinatlas.tsv文件'
                logger.error(f"【HPA下载】✗ {result['error']}")

        except requests.exceptions.Timeout:
            result['error'] = '下载超时（超过5分钟），请检查网络连接后刷新页面重试'
            logger.error(f"【HPA下载】✗ 下载超时")
        except requests.exceptions.ConnectionError:
            result['error'] = '网络连接错误，请检查网络后刷新页面重试'
            logger.error(f"【HPA下载】✗ 网络连接错误")
        except Exception as e:
            result['error'] = f'下载失败: {str(e)}'
            logger.error(f"【HPA下载】✗ {result['error']}", exc_info=True)
        
        return result

# ==================== HPA基因自动补全服务（基于Gene synonym）====================
class HPAGeneAutocompleteService:
    """基于HPA proteinatlas.tsv的Gene synonym列的基因自动补全服务"""
    
    def __init__(self, hpa_manager: 'HPADataManager' = None):
        self.hpa_manager = hpa_manager
        self.search_index = {}  # {normalized_name: {gene_symbol, synonyms, ensembl_id}}
        self.gene_data_cache = {}  # {gene_symbol_upper: gene_data_dict}
        self._build_search_index()
    
    def _build_search_index(self):
        """从proteinatlas.tsv构建基因搜索索引"""
        data_file = None
        
        # 尝试获取HPA数据文件路径
        if self.hpa_manager and hasattr(self.hpa_manager, 'data_file'):
            data_file = self.hpa_manager.data_file
        else:
            local_dir = os.path.join(os.path.expanduser("~"), ".hpa_data")
            data_file = os.path.join(local_dir, "proteinatlas.tsv")
        
        if not data_file or not os.path.exists(data_file):
            logger.warning("HPA数据文件不存在，基因自动补全将不可用")
            return
        
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter='\t')
                for row in reader:
                    gene_symbol = row.get('Gene name', '').strip()
                    ensembl_id = row.get('Gene', '').strip()
                    synonyms_str = row.get('Gene synonym', '').strip()
                    
                    # 如果Gene name为空，尝试使用Gene列（Ensembl ID）作为主键
                    if not gene_symbol and ensembl_id:
                        gene_symbol = ensembl_id
                    
                    if not gene_symbol:
                        continue
                    
                    gene_key = gene_symbol.upper()
                    
                    # 存储基因数据
                    self.gene_data_cache[gene_key] = {
                        'gene_symbol': gene_symbol,
                        'ensembl_id': ensembl_id,
                        'synonyms': synonyms_str,
                        'chromosome': row.get('Chromosome', ''),
                        'position': row.get('Position', ''),
                        'uniprot': row.get('Uniprot', ''),
                        'description': row.get('Gene description', '')
                    }
                    
                    # 索引Gene name列（主要基因名）
                    if gene_symbol and gene_symbol != ensembl_id:
                        self._add_to_index(gene_symbol, gene_symbol, ensembl_id, 'primary')
                    
                    # 索引Gene列（Ensembl ID）
                    if ensembl_id:
                        self._add_to_index(ensembl_id, gene_symbol, ensembl_id, 'ensembl')
                    
                    # 索引所有synonyms
                    if synonyms_str:
                        for syn in synonyms_str.split(','):
                            syn_clean = syn.strip()
                            if syn_clean and syn_clean.upper() != gene_symbol.upper():
                                self._add_to_index(syn_clean, gene_symbol, ensembl_id, 'synonym')
                    
        except Exception as e:
            logger.error(f"构建HPA基因索引失败: {e}")
    
    def _add_to_index(self, name: str, gene_symbol: str, ensembl_id: str, name_type: str):
        """添加名称到搜索索引"""
        norm = self._normalize(name)
        if norm and norm not in self.search_index:
            self.search_index[norm] = {
                'gene_symbol': gene_symbol,
                'ensembl_id': ensembl_id,
                'name_type': name_type,
                'original_name': name
            }
    
    def _normalize(self, name: str) -> str:
        """标准化名称用于搜索"""
        if not name:
            return ""
        return name.upper().replace('-', '').replace(' ', '').replace('_', '')
    
    def rebuild_index(self) -> bool:
        """重新构建搜索索引（数据下载后调用）"""
        self.search_index = {}
        self.gene_data_cache = {}
        self._build_search_index()
        return len(self.search_index) > 0
    
    def is_index_ready(self) -> bool:
        """检查索引是否已构建"""
        return len(self.search_index) > 0
    
    def get_suggestions(self, query: str, limit: int = 8) -> List[Dict]:
        """
        获取基因建议（输入2个字符以上显示建议）
        匹配逻辑：精确匹配 > 前缀匹配 > 包含匹配 > 模糊匹配
        """
        if not query or len(query) < 2:
            return []
        
        query_norm = self._normalize(query)
        if not query_norm:
            return []
        
        matches = []
        seen_genes = set()
        
        # 1. 精确匹配
        if query_norm in self.search_index:
            info = self.search_index[query_norm]
            gene_symbol = info['gene_symbol']
            if gene_symbol.upper() not in seen_genes:
                matches.append({
                    'display_name': gene_symbol,
                    'gene_symbol': gene_symbol,
                    'match_type': 'exact',
                    'matched_name': info['original_name'],
                    'name_type': info['name_type'],
                    'score': 100
                })
                seen_genes.add(gene_symbol.upper())
        
        # 2. 前缀匹配
        for norm, info in self.search_index.items():
            gene_symbol = info['gene_symbol']
            if gene_symbol.upper() not in seen_genes and norm.startswith(query_norm):
                matches.append({
                    'display_name': gene_symbol,
                    'gene_symbol': gene_symbol,
                    'match_type': 'prefix',
                    'matched_name': info['original_name'],
                    'name_type': info['name_type'],
                    'score': 80 - len(norm)
                })
                seen_genes.add(gene_symbol.upper())
                if len(seen_genes) >= limit * 2:
                    break
        
        # 3. 包含匹配
        for norm, info in self.search_index.items():
            gene_symbol = info['gene_symbol']
            if gene_symbol.upper() not in seen_genes and query_norm in norm:
                matches.append({
                    'display_name': gene_symbol,
                    'gene_symbol': gene_symbol,
                    'match_type': 'substring',
                    'matched_name': info['original_name'],
                    'name_type': info['name_type'],
                    'score': 50
                })
                seen_genes.add(gene_symbol.upper())
                if len(seen_genes) >= limit * 2:
                    break
        
        # 4. 模糊匹配（查询3字符以上）
        if len(query_norm) >= 3:
            for norm, info in self.search_index.items():
                gene_symbol = info['gene_symbol']
                if gene_symbol.upper() not in seen_genes:
                    sim = difflib.SequenceMatcher(None, query_norm, norm).ratio()
                    if sim > 0.6:
                        matches.append({
                            'display_name': gene_symbol,
                            'gene_symbol': gene_symbol,
                            'match_type': 'fuzzy',
                            'matched_name': info['original_name'],
                            'name_type': info['name_type'],
                            'score': int(sim * 40)
                        })
                        seen_genes.add(gene_symbol.upper())
                        if len(seen_genes) >= limit * 2:
                            break
        
        # 排序并限制数量
        matches.sort(key=lambda x: (x['score'], x['display_name']), reverse=True)
        return matches[:limit]
    
    def get_gene_info(self, gene_symbol: str) -> Optional[Dict]:
        """获取基因的完整信息"""
        return self.gene_data_cache.get(gene_symbol.upper())
    
    def is_valid_gene(self, gene_symbol: str) -> bool:
        """检查基因是否有效"""
        return gene_symbol.upper() in self.gene_data_cache


# ==================== HPA基因详细信息提取服务 ====================
class HPAGeneDetailService:
    """从proteinatlas.tsv提取基因详细信息"""
    
    def __init__(self, hpa_manager: 'HPADataManager' = None):
        self.hpa_manager = hpa_manager
        self.data_file = self._get_data_file()
    
    def _get_data_file(self) -> Optional[str]:
        """获取数据文件路径"""
        if self.hpa_manager and hasattr(self.hpa_manager, 'data_file'):
            return self.hpa_manager.data_file
        local_dir = os.path.join(os.path.expanduser("~"), ".hpa_data")
        return os.path.join(local_dir, "proteinatlas.tsv")
    
    def get_gene_details(self, gene_symbol: str) -> Optional[Dict]:
        """获取基因详细信息 - 在Gene name、Gene和Gene synonym三列搜索"""
        if not self.data_file:
            logger.error("HPA数据文件路径未设置")
            return {'error': '数据文件路径未设置', 'data_file': None}
        
        if not os.path.exists(self.data_file):
            logger.error(f"HPA数据文件不存在: {self.data_file}")
            return {'error': f'数据文件不存在: {self.data_file}', 'data_file': self.data_file}
        
        # 使用校正后的基因名（大写并去除空格）
        gene_symbol_upper = gene_symbol.upper().strip()
        logger.info(f"查询HPA基因: {gene_symbol_upper}, 数据文件: {self.data_file}")
        
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter='\t')
                row_count = 0
                matched_row = None
                
                for row in reader:
                    row_count += 1
                    
                    # 1. 在 Gene name 列搜索
                    gene_name = row.get('Gene name', '').upper().strip()
                    if gene_name == gene_symbol_upper:
                        logger.info(f"在Gene name列找到基因 {gene_symbol_upper} 在第 {row_count} 行")
                        matched_row = row
                        break
                    
                    # 2. 在 Gene 列（Ensembl ID）搜索
                    ensembl_id = row.get('Gene', '').upper().strip()
                    if ensembl_id == gene_symbol_upper:
                        logger.info(f"在Gene(Ensembl)列找到基因 {gene_symbol_upper} 在第 {row_count} 行")
                        matched_row = row
                        break
                    
                    # 3. 在 Gene synonym 列搜索（同义词用逗号分隔）
                    synonyms_str = row.get('Gene synonym', '')
                    if synonyms_str:
                        synonyms = [s.strip().upper() for s in synonyms_str.split(',')]
                        if gene_symbol_upper in synonyms:
                            logger.info(f"在Gene synonym列找到基因 {gene_symbol_upper} 在第 {row_count} 行")
                            matched_row = row
                            break
                
                if matched_row:
                    return self._extract_gene_data(matched_row)
                else:
                    logger.warning(f"扫描了 {row_count} 行，未找到基因: {gene_symbol_upper}")
                    return {'error': f'未找到基因: {gene_symbol}', 'data_file': self.data_file, 'rows_scanned': row_count}
                    
        except Exception as e:
            logger.error(f"获取基因详情失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {'error': f'读取数据文件失败: {str(e)}', 'data_file': self.data_file}
    
    def _extract_gene_data(self, row: Dict) -> Dict:
        """从行数据中提取基因信息"""
        
        # a. Ensembl ID 和链接
        ensembl_id = row.get('Gene', '')
        
        # b. Uniprot ID 和链接
        uniprot_id = row.get('Uniprot', '')
        
        # c. 基因组位置
        chromosome = row.get('Chromosome', '')
        position = row.get('Position', '')
        
        # d. 蛋白定位与功能
        subcellular_main = row.get('Subcellular main location', '')
        subcellular_add = row.get('Subcellular additional location', '')
        secretome_loc = row.get('Secretome location', '')
        secretome_func = row.get('Secretome function', '')
        biological_process = row.get('Biological process', '')
        molecular_func = row.get('Molecular function', '')
        disease_involvement = row.get('Disease involvement', '')
        
        # e. RNA表达与定位
        rna_tissue_specificity = row.get('RNA tissue specificity', '')
        rna_tissue_specific_ntpm = row.get('RNA tissue specific nTPM', '')
        
        # f. 抗体推荐
        antibody = row.get('Antibody', '')
        
        # g. RNA在不同样本中的表达
        rna_single_cell = {
            'specificity': row.get('RNA single cell type specificity', ''),
            'specific_ncpm': row.get('RNA single cell type specific nCPM', '')
        }
        rna_cancer = {
            'specificity': row.get('RNA cancer specificity', ''),
            'specific_ptpm': row.get('RNA cancer specific pTPM', '')
        }
        rna_blood = {
            'specificity': row.get('RNA blood cell specificity', ''),
            'specific_ntpm': row.get('RNA blood cell specific nTPM', '')
        }
        
        return {
            'gene_symbol': row.get('Gene name', ''),
            'ensembl_id': ensembl_id,
            'ensembl_url': f"https://www.ensembl.org/Homo_sapiens/Gene/Summary?g={ensembl_id}" if ensembl_id else '',
            'uniprot_id': uniprot_id,
            'uniprot_url': f"https://www.uniprot.org/uniprotkb/{uniprot_id}" if uniprot_id else '',
            'genome_location': f"{chromosome}: {position}" if chromosome and position else '',
            'chromosome': chromosome,
            'position': position,
            'protein_localization': {
                'subcellular_main': subcellular_main,
                'subcellular_additional': subcellular_add,
                'secretome_location': secretome_loc,
                'secretome_function': secretome_func
            },
            'protein_function': {
                'biological_process': biological_process,
                'molecular_function': molecular_func,
                'disease_involvement': disease_involvement
            },
            'rna_expression': {
                'tissue_specificity': rna_tissue_specificity,
                'tissue_specific_ntpm': rna_tissue_specific_ntpm
            },
            'antibody': {
                'name': antibody,
                'hpa_search_url': f"https://www.proteinatlas.org/search/{antibody.replace(' ', '%20')}" if antibody else '',
                'hpa_gene_url': f"https://www.proteinatlas.org/{ensembl_id}" if ensembl_id else ''
            },
            'rna_distribution': {
                'tissue': {
                    'specificity': rna_tissue_specificity,
                    'specific_ntpm': rna_tissue_specific_ntpm
                },
                'single_cell': rna_single_cell,
                'cancer': rna_cancer,
                'blood': rna_blood
            }
        }

class APIConfig:
    @staticmethod
    def get_ncbi_credentials():
        user_email = st.session_state.get('ncbi_email_input', '').strip()
        user_key = st.session_state.get('ncbi_key_input', '').strip()

        try:
            secret_email = st.secrets.get("NCBI_EMAIL", "")
            secret_key = st.secrets.get("NCBI_API_KEY", "")
        except:
            secret_email = ""
            secret_key = ""

        email = user_email if user_email else secret_email
        api_key = user_key if user_key else secret_key

        if not email or email == "user@example.com":
            if not secret_email or secret_email == "user@example.com":
                return None, None, "API失效，需要填入有效API"

        return email, api_key, None

    @staticmethod
    def get_qwen_api_key():
        """修复版API Key获取 - 确保正确检查session_state和secrets"""
        # 优先检查session_state中的用户输入
        user_key = st.session_state.get('qwen_key_input', '').strip()

        # 检查secrets
        secret_key = None
        try:
            secret_key = st.secrets.get("DASHSCOPE_API_KEY") or st.secrets.get("QWEN_API_KEY")
        except Exception as e:
            logger.warning(f"读取secrets失败: {e}")

        # 优先使用用户输入的key
        final_key = user_key if user_key else (secret_key if secret_key else None)

        if final_key:
            logger.info(f"Qwen API Key已配置")

        return final_key

# ==================== 频率限制器 ====================
class APIRateLimiter:
    def __init__(self, requests_per_second: float = 3.0):
        self.min_interval = 1.0 / requests_per_second
        self.last_request_time = 0

    def wait(self):
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_request_time = time.time()

ncbi_limiter = APIRateLimiter(3.0)

# ==================== 重试装饰器（新增）====================
def retry_on_failure(max_retries=3, delay=1.0):
    """重试装饰器 - 用于网络请求"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"{func.__name__} 失败 (尝试 {attempt+1}/{max_retries}): {e}")
                        time.sleep(delay * (attempt + 1))
                    else:
                        logger.error(f"{func.__name__} 最终失败: {e}")
                        raise
            return None
        return wrapper
    return decorator

# ==================== NCBI客户端 ====================
class NCBIClient:
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def __init__(self, email: str, api_key: Optional[str] = None):
        self.email = email
        self.api_key = api_key

    def _make_request(self, endpoint: str, params: Dict, retmode: str = "json") -> Optional[Dict]:
        ncbi_limiter.wait()
        params.update({'tool': 'LentivirusAssessment', 'email': self.email})
        if self.api_key:
            params['api_key'] = self.api_key

        url = f"{self.BASE_URL}/{endpoint}"
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json() if retmode == "json" else response.text
        except requests.exceptions.HTTPError as e:
            logger.error(f"NCBI HTTP error: {e}, Status: {response.status_code}")
            # 检查是否是认证问题
            if response.status_code == 403:
                logger.error("NCBI API 访问被拒绝，请检查 email 和 API key 配置")
            elif response.status_code == 429:
                logger.error("NCBI API 请求过于频繁，请添加 API key 以提高限额")
            return None
        except requests.exceptions.Timeout as e:
            logger.error(f"NCBI request timeout: {e}")
            return None
        except requests.exceptions.ConnectionError as e:
            logger.error(f"NCBI connection error: {e}")
            return None
        except Exception as e:
            logger.error(f"NCBI request failed: {type(e).__name__}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    @safe_cache_data
    def fetch_gene_data(_self, gene_name: str, organism: str) -> Tuple[Dict, List[Dict]]:
        search_params = {
            'db': 'gene',
            'term': f"{gene_name}[Gene] AND {organism}[Organism]",
            'retmode': 'json',
            'retmax': 1
        }
        result = _self._make_request('esearch.fcgi', search_params)
        if not result:
            return {}, []

        gene_ids = result.get('esearchresult', {}).get('idlist', [])
        if not gene_ids:
            return {}, []

        gene_id = gene_ids[0]

        summary_params = {'db': 'gene', 'id': gene_id, 'retmode': 'json'}
        result = _self._make_request('esummary.fcgi', summary_params)
        if not result:
            return {}, {}

        summary = result.get('result', {}).get(gene_id, {})
        # 使用NCBI返回的基因符号（优先Symbol，其次是NomenclatureName，最后是name）
        # symbol 是基因符号如 "WDR83"，name 是描述性名称如 "WD repeat domain 83"
        official_name = summary.get('symbol', '') or summary.get('genesymbol', '') or summary.get('nomenclaturename', '') or summary.get('name', '') or gene_name
        gene_info = {
            'id': gene_id,
            'name': official_name,
            'description': summary.get('description', ''),
            'organism': organism,
            'summary': summary.get('summary', '')
        }

        transcripts = _self._fetch_transcripts(gene_id)
        return gene_info, transcripts

    @retry_on_failure(max_retries=3, delay=1.0)
    def _fetch_transcripts(self, gene_id: str) -> List[Dict]:
        """使用NCBI E-utilities获取转录本（带重试和限流）"""
        try:
            # 步骤1：搜索nuccore数据库 - 尝试多种搜索策略
            ids = []
            debug_info = []
            
            # 策略列表：从最精确到最宽泛
            search_terms = [
                # 策略1：使用GeneID和RefSeq过滤
                f"{gene_id}[GeneID] AND refseq[Filter]",
                # 策略2：直接GeneID搜索（不加限制，获取所有相关核酸序列）
                f"{gene_id}[GeneID]",
                # 策略3：尝试不同格式的GeneID
                f"{gene_id}[uid]",
            ]
            
            for term in search_terms:
                search_params = {
                    'db': 'nuccore',
                    'term': term,
                    'retmode': 'json',
                    'retmax': 50,  # 增加返回数量
                    'sort': 'accession'
                }
                result = self._make_request('esearch.fcgi', search_params)
                if result:
                    term_ids = result.get('esearchresult', {}).get('idlist', [])
                    debug_info.append(f"策略 '{term[:40]}...': 找到 {len(term_ids)} 个ID")
                    if term_ids:
                        ids.extend(term_ids)
                        logger.info(f"找到 {len(term_ids)} 个转录本 (查询: {term[:50]}...)")
                else:
                    debug_info.append(f"策略 '{term[:40]}...': 请求失败 (None)")
            
            # 去重
            ids = list(dict.fromkeys(ids))  # 保持顺序去重
            
            if not ids:
                logger.warning(f"基因ID {gene_id} 未找到任何转录本，尝试elink备选方案...")
                # 备选方案：使用elink从gene链接到nuccore
                try:
                    elink_params = {
                        'dbfrom': 'gene',
                        'db': 'nuccore',
                        'id': gene_id,
                        'retmode': 'json',
                        'linkname': 'gene_nuccore_refseqrna'
                    }
                    elink_result = self._make_request('elink.fcgi', elink_params)
                    if elink_result:
                        linksets = elink_result.get('linksets', [])
                        for linkset in linksets:
                            linksetdbs = linkset.get('linksetdbs', [])
                            for linksetdb in linksetdbs:
                                if linksetdb.get('dbto') == 'nuccore':
                                    new_ids = linksetdb.get('links', [])
                                    if new_ids:
                                        ids.extend(new_ids)
                                        debug_info.append(f"elink备选方案: 找到 {len(new_ids)} 个ID")
                    if not ids:
                        debug_info.append("elink备选方案: 未找到任何转录本")
                except Exception as e:
                    debug_info.append(f"elink备选方案失败: {e}")
                
                if not ids:
                    self._last_transcript_debug = debug_info
                    return []

            # 步骤2：获取详细信息
            summary_params = {
                'db': 'nuccore',
                'id': ','.join(ids[:20]),  # 限制数量避免请求过大
                'retmode': 'json'
            }
            summary_result = self._make_request('esummary.fcgi', summary_params)
            if not summary_result:
                logger.warning("NCBI esummary 请求失败")
                return []

            docs = summary_result.get('result', {})
            transcripts = []
            for uid in ids[:20]:  # 限制处理数量
                try:
                    doc = docs.get(str(uid), {})
                    acc = doc.get('accessionversion', '')
                    slen = doc.get('slen', 0)
                    title = str(doc.get('title', ''))

                    # 更宽松的筛选：包含 RefSeq mRNA 或 title 包含 gene 关键词
                    is_refseq = acc.startswith('NM_') or acc.startswith('XM_') or acc.startswith('NR_')
                    is_mrna = 'mRNA' in title or 'transcript' in title.lower()
                    
                    if not acc:
                        continue
                        
                    # 优先保留 RefSeq 转录本
                    if is_refseq or is_mrna:
                        tx_type = 'NM' if acc.startswith('NM_') else ('XM' if acc.startswith('XM_') else 'OTHER')
                        status = 'REVIEWED' if tx_type == 'NM' else ('PREDICTED' if tx_type == 'XM' else 'UNKNOWN')

                        transcripts.append({
                            'id': acc,
                            'length': int(slen) if slen else 0,
                            'type': tx_type,
                            'status': status,
                            'title': title[:100]
                        })
                except Exception as e:
                    logger.warning(f"解析转录本 {uid} 失败: {e}")
                    continue

            # 排序：NM优先，然后按长度降序
            transcripts.sort(key=lambda x: (0 if x['type'] == 'NM' else (1 if x['type'] == 'XM' else 2), -x['length']))
            
            if not transcripts:
                logger.warning(f"解析后未发现有效RefSeq转录本，原始ID数: {len(ids)}")
                
            return transcripts

        except Exception as e:
            logger.error(f"获取转录本失败: {e}")
            return []

        except Exception as e:
            logger.error(f"获取转录本失败: {e}")
            return []

    def search_gene_function_literature(self, gene_name: str, query_type: str) -> List[Dict]:
        query_map = {
            'general': [
                f"{gene_name} function protein characterization",
                f"{gene_name} biological role pathway"
            ],
            'overexpression': [
                f"{gene_name} overexpression cell line phenotype",
                f"{gene_name} ectopic expression tumor",
                f"{gene_name} transgenic mouse model"
            ],
            'knockdown': [
                f"{gene_name} siRNA knockdown phenotype",
                f"{gene_name} shRNA silencing effect"
            ],
            'knockout': [
                f"{gene_name} knockout mouse phenotype",
                f"{gene_name} CRISPR knockout cell",
                f"{gene_name} gene deletion embryonic lethal"
            ]
        }

        queries = query_map.get(query_type, [f"{gene_name}"])
        all_papers = []
        seen_pmids = set()
        
        logger.info(f"【文献检索】开始检索 {gene_name} 的 {query_type} 文献，使用 {len(queries)} 个查询词")

        for i, query in enumerate(queries):
            try:
                logger.info(f"【文献检索】查询 {i+1}/{len(queries)}: {query}")
                
                search_params = {
                    'db': 'pubmed',
                    'term': query,
                    'retmode': 'json',
                    'retmax': 8,
                    'sort': 'relevance'
                }
                result = self._make_request('esearch.fcgi', search_params)
                
                if not result:
                    logger.warning(f"【文献检索】查询 '{query}' 返回空结果")
                    continue

                pmids = result.get('esearchresult', {}).get('idlist', [])
                logger.info(f"【文献检索】查询 '{query}' 找到 {len(pmids)} 个PMID: {pmids[:5]}")
                
                new_pmids = [p for p in pmids if p not in seen_pmids]
                if not new_pmids:
                    logger.info(f"【文献检索】查询 '{query}' 无新PMID")
                    continue

                fetch_params = {'db': 'pubmed', 'id': ','.join(new_pmids), 'retmode': 'json'}
                result = self._make_request('esummary.fcgi', fetch_params)
                
                if not result:
                    logger.warning(f"【文献检索】获取详情失败 for PMIDs: {new_pmids}")
                    continue

                docs = result.get('result', {})
                success_count = 0
                
                for pmid in new_pmids:
                    try:
                        doc = docs.get(pmid, {})
                        title = doc.get('title', '')
                        abstract = doc.get('abstract', '') or doc.get('sorttitle', '')

                        if not title:
                            logger.warning(f"【文献检索】PMID {pmid} 无标题")
                            continue

                        all_papers.append({
                            'pmid': str(pmid),
                            'title': html.escape(str(title)[:300]),
                            'abstract': html.escape(str(abstract)[:800]) if abstract else "[无摘要]",
                            'authors': doc.get('authors', []),
                            'source': doc.get('source', ''),
                            'pubdate': doc.get('pubdate', '')[:4] if doc.get('pubdate') else '',
                            'url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                        })
                        seen_pmids.add(pmid)
                        success_count += 1
                    except Exception as e:
                        logger.error(f"【文献检索】处理 PMID {pmid} 失败: {e}")
                        continue
                
                logger.info(f"【文献检索】查询 '{query}' 成功获取 {success_count} 篇文献")

            except Exception as e:
                logger.error(f"【文献检索】查询 '{query}' 失败: {e}")
                continue
        
        logger.info(f"【文献检索】{gene_name} {query_type} 总计获取 {len(all_papers)} 篇文献")
        return all_papers

    def search_gene_property_literature(self, gene_name: str, property_type: str) -> List[Dict]:
        query_map = {
            'essential': [
                f"{gene_name} knockout lethal",
                f"{gene_name} knockout cell death essential",
                f"{gene_name} CRISPR knockout viability loss"
            ],
            'toxic': [
                f"{gene_name} overexpression cytotoxic",
                f"{gene_name} overexpression apoptosis cell death",
                f"{gene_name} overexpression growth inhibition"
            ],
            'antiviral': [
                f"{gene_name} interferon antiviral innate immunity",
                f"{gene_name} virus replication restriction factor",
                f"{gene_name} ISG interferon stimulated",
                f"{gene_name} IFITM antiviral",
                f"{gene_name} transcription factor antiviral gene",
                f"{gene_name} regulates interferon stimulated gene",
                f"{gene_name} host factor virus infection",
                f"{gene_name} STING MDA5 RIG-I pathway",
                f"{gene_name} influenza HIV SARS-CoV-2",
                f"{gene_name} virus susceptibility resistance",
                f"{gene_name} viral infection immune response",
                f"{gene_name} antiviral defense mechanism",
                f"{gene_name} innate immunity virus",
                f"{gene_name} infection response",
                f"{gene_name} virus entry replication"
            ]
        }

        queries = query_map.get(property_type, [f"{gene_name} function"])
        all_papers = []
        seen_pmids = set()

        for query in queries:
            try:
                search_params = {
                    'db': 'pubmed',
                    'term': query,
                    'retmode': 'json',
                    'retmax': 5,
                    'sort': 'relevance'
                }
                result = self._make_request('esearch.fcgi', search_params)
                if not result:
                    continue

                pmids = result.get('esearchresult', {}).get('idlist', [])
                new_pmids = [p for p in pmids if p not in seen_pmids]
                if not new_pmids:
                    continue

                fetch_params = {'db': 'pubmed', 'id': ','.join(new_pmids), 'retmode': 'json'}
                result = self._make_request('esummary.fcgi', fetch_params)
                if not result:
                    continue

                docs = result.get('result', {})
                for pmid in new_pmids:
                    try:
                        doc = docs.get(pmid, {})
                        title = doc.get('title', '')
                        abstract = doc.get('abstract', '') or ''

                        if not title:
                            continue

                        all_papers.append({
                            'pmid': str(pmid),
                            'title': html.escape(str(title)[:300]),
                            'abstract': html.escape(str(abstract)[:800]) if abstract else "[无摘要]",
                            'query': query,
                            'url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                        })
                        seen_pmids.add(pmid)
                    except Exception:
                        continue

            except Exception as e:
                logger.error(f"Literature search error: {e}")
                continue

        return all_papers

    def search_cell_lentivirus_params(self, cell_name: str) -> List[Dict]:
        params_list = []
        try:
            queries = [f"{cell_name} lentivirus MOI", f"{cell_name} lentiviral transduction"]
            for query in queries:
                search_params = {
                    'db': 'pubmed',
                    'term': query,
                    'retmode': 'json',
                    'retmax': 5
                }
                result = self._make_request('esearch.fcgi', search_params)
                if result:
                    pmids = result.get('esearchresult', {}).get('idlist', [])
                    for pmid in pmids:
                        params_list.append({
                            'pmid': pmid,
                            'source': 'PubMed',
                            'note': '需查阅全文获取MOI、温度、离心参数',
                            'url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                        })
            return params_list if params_list else []
        except Exception as e:
            logger.error(f"Cell params search error: {e}")
            return []

    def search_cell_transfection(self, cell_name: str) -> List[Dict]:
        try:
            query = f"{cell_name} siRNA transfection electroporation"
            search_params = {
                'db': 'pubmed',
                'term': query,
                'retmode': 'json',
                'retmax': 5
            }
            result = self._make_request('esearch.fcgi', search_params)
            if not result:
                return []

            pmids = result.get('esearchresult', {}).get('idlist', [])
            return [{'pmid': pmid, 'note': '需查阅全文获取转染条件', 'url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"} for pmid in pmids]

        except Exception as e:
            logger.error(f"Transfection search error: {e}")
            return []

    def search_same_cell_gene_studies(self, gene_name: str, cell_name: str) -> List[Dict]:
        try:
            query = f"{gene_name} {cell_name} overexpression knockdown"
            search_params = {
                'db': 'pubmed',
                'term': query,
                'retmode': 'json',
                'retmax': 10
            }
            result = self._make_request('esearch.fcgi', search_params)
            if not result:
                return []

            pmids = result.get('esearchresult', {}).get('idlist', [])
            studies = []

            for pmid in pmids:
                fetch_params = {'db': 'pubmed', 'id': pmid, 'retmode': 'json'}
                result = self._make_request('esummary.fcgi', fetch_params)
                if result:
                    doc = result.get('result', {}).get(pmid, {})
                    studies.append({
                        'pmid': pmid,
                        'title': doc.get('title', ''),
                        'journal': doc.get('fulljournalname', ''),
                        'year': doc.get('pubdate', '')[:4] if doc.get('pubdate') else '',
                        'url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                    })

            return studies

        except Exception as e:
            logger.error(f"Same cell/gene search error: {e}")
            return []

    def search_cell_culture_literature(self, cell_name: str) -> List[Dict]:
        try:
            queries = [
                f"{cell_name} cell culture protocol",
                f"{cell_name} culture medium serum",
                f"{cell_name} matrigel coating",
                f"{cell_name} feeder layer",
                f"{cell_name} culture difficulty"
            ]

            all_papers = []
            seen_pmids = set()

            for query in queries:
                search_params = {
                    'db': 'pubmed',
                    'term': query,
                    'retmode': 'json',
                    'retmax': 3,
                    'sort': 'relevance'
                }
                result = self._make_request('esearch.fcgi', search_params)
                if not result:
                    continue

                pmids = result.get('esearchresult', {}).get('idlist', [])
                new_pmids = [p for p in pmids if p not in seen_pmids]
                if not new_pmids:
                    continue

                fetch_params = {'db': 'pubmed', 'id': ','.join(new_pmids), 'retmode': 'json'}
                result = self._make_request('esummary.fcgi', fetch_params)
                if not result:
                    continue

                docs = result.get('result', {})
                for pmid in new_pmids:
                    try:
                        doc = docs.get(pmid, {})
                        title = doc.get('title', '')
                        abstract = doc.get('abstract', '') or ''

                        if not title:
                            continue

                        all_papers.append({
                            'pmid': str(pmid),
                            'title': html.escape(str(title)[:300]),
                            'abstract': html.escape(str(abstract)[:500]) if abstract else "[无摘要]",
                            'url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                        })
                        seen_pmids.add(pmid)
                    except Exception:
                        continue

            return all_papers

        except Exception as e:
            logger.error(f"Cell culture literature search error: {e}")
            return []

# ==================== 数据模型 ====================
@dataclass
class HardRuleCheck:
    rule_name: str
    passed: bool
    reason: str
    source: str
    pmid: Optional[str] = None
    pmid_list: List[str] = field(default_factory=list)
    overrideable: bool = False
    evidence_papers: List[Dict] = field(default_factory=list)
    check_level: str = "core"

# ==================== 混合硬性规则引擎 ====================
class HybridHardRulesEngine:
    def __init__(self, ncbi_client: NCBIClient, ai_client: Optional[AIAnalysisClient] = None):
        self.ncbi = ncbi_client
        self.ai = ai_client

    def check_all(self, gene_name: str, transcripts: List[Dict],
                  experiment_type: str) -> Tuple[bool, List[HardRuleCheck], Dict]:
        checks = []
        evidence_summary = {
            'essential_checked': False,
            'toxic_checked': False,
            'antiviral_checked': False,
            'core_hits': [],
            'literature_hits': [],
            'ai_analyzed': False
        }

        if experiment_type.lower() == 'overexpression':
            check = self._check_vector_capacity(gene_name, transcripts)
            checks.append(check)

        if experiment_type.lower() == 'knockout':
            core_result = CoreDatabases.check_gene(gene_name, 'essential')
            if core_result:
                pmid, source, desc = core_result
                check = HardRuleCheck(
                    rule_name="必需基因检查（核心数据库）",
                    passed=False,
                    reason=f"{gene_name}是{source}（{desc}），敲除可能导致细胞死亡",
                    source=f"DepMap数据库（{source}）",
                    pmid=pmid,
                    overrideable=False,
                    check_level="core"
                )
                checks.append(check)
                evidence_summary['essential_checked'] = True
                evidence_summary['core_hits'].append('essential')
            else:
                lit_check = self._check_by_literature(gene_name, 'essential')
                checks.append(lit_check)
                if not lit_check.passed:
                    evidence_summary['essential_checked'] = True
                    evidence_summary['literature_hits'].append('essential')

        if experiment_type.lower() == 'overexpression':
            core_toxic = CoreDatabases.check_gene(gene_name, 'toxic')
            if core_toxic:
                pmid, source, desc = core_toxic
                check = HardRuleCheck(
                    rule_name="毒性基因检查（核心数据库）",
                    passed=False,
                    reason=f"{gene_name}是{source}（{desc}），过表达可能导致细胞死亡",
                    source=f"毒性基因数据库（{source}）",
                    pmid=pmid,
                    overrideable=False,
                    check_level="core"
                )
                checks.append(check)
                evidence_summary['toxic_checked'] = True
                evidence_summary['core_hits'].append('toxic')
            else:
                lit_check = self._check_by_literature(gene_name, 'toxic')
                checks.append(lit_check)
                if not lit_check.passed:
                    evidence_summary['toxic_checked'] = True
                    evidence_summary['literature_hits'].append('toxic')

        core_antiviral = CoreDatabases.check_gene(gene_name, 'antiviral')
        if core_antiviral:
            pmid, source, desc = core_antiviral
            check = HardRuleCheck(
                rule_name="抗病毒基因检查（核心数据库）",
                passed=False,
                reason=f"{gene_name}是{source}（{desc}），过表达可能抑制慢病毒包装",
                source=f"ISG数据库（{source}）",
                pmid=pmid,
                overrideable=False,
                check_level="core"
            )
            checks.append(check)
            evidence_summary['antiviral_checked'] = True
            evidence_summary['core_hits'].append('antiviral')
        else:
            lit_check = self._check_by_literature_enhanced(gene_name, 'antiviral')
            checks.append(lit_check)
            if not lit_check.passed:
                evidence_summary['antiviral_checked'] = True
                evidence_summary['literature_hits'].append('antiviral')

        if self.ai and self.ai.api_key:
            evidence_summary['ai_analyzed'] = True

        return all(c.passed for c in checks), checks, evidence_summary

    def _check_vector_capacity(self, gene_name: str, transcripts: List[Dict]) -> HardRuleCheck:
        valid_lengths = [t.get('length', 0) for t in transcripts if t.get('length', 0) > 0]

        if not valid_lengths:
            return HardRuleCheck(
                rule_name="载体容量检查（过表达）",
                passed=True,
                reason="转录本长度信息暂不可获得",
                source="NCBI nuccore数据库",
                overrideable=True,
                check_level="core"
            )

        max_length = max(valid_lengths)
        if max_length <= 2000:
            return HardRuleCheck(
                rule_name="载体容量检查（过表达）",
                passed=True,
                reason=f"转录本长度 {max_length}bp ≤2000bp，适合标准过表达",
                source="NCBI nuccore数据库",
                overrideable=True,
                check_level="core"
            )
        else:
            return HardRuleCheck(
                rule_name="载体容量检查（过表达）",
                passed=True,
                reason=f"转录本长度 {max_length}bp 超过2000bp，属于长序列过表达",
                source="NCBI nuccore数据库",
                pmid="PMID:15819909",
                overrideable=True,
                check_level="core"
            )

    def _check_by_literature(self, gene_name: str, check_type: str) -> HardRuleCheck:
        papers = self.ncbi.search_gene_property_literature(gene_name, check_type)

        if not papers:
            type_names = {'essential': '必需性', 'toxic': '毒性', 'antiviral': '抗病毒功能'}
            return HardRuleCheck(
                rule_name=f"{type_names[check_type]}检查（文献补充）",
                passed=True,
                reason=f"核心数据库未收录，且未检索到相关文献",
                source="核心数据库+PubMed（零结果）",
                overrideable=True,
                check_level="literature"
            )

        type_names = {'essential': '必需性', 'toxic': '毒性', 'antiviral': '抗病毒功能'}

        if check_type == 'antiviral':
            # 逆转录病毒特异性抗病毒因子检测（慢病毒包装相关）
            antiviral_keywords = {
                'retrovirus_specific': [
                    'restrict hiv', 'hiv restriction', 'lentivirus restriction',
                    'retrovirus restriction', 'restrict retrovirus',
                    'apobec', 'trim5', 'tetherin', 'bst2', 'cd317',
                    'samhd1', 'mx2', 'mx1', 'serinc', 'marchf',
                    'cypa', 'cyclophilin a'
                ],
                'host_factor_retrovirus': [
                    'hiv replication', 'lentivirus replication', 'retroviral replication',
                    'viral entry hiv', 'hiv entry receptor', 'co-receptor',
                    'reverse transcription', 'integration', 'proviral',
                    'gag protein', 'env protein', 'pol protein'
                ],
                'isg_retrovirus': [
                    'isg', 'interferon stimulated gene',
                    'ifitm', 'isg15', 'oas', 'pkr', 'viperin',
                    'rsad2', 'ifit', 'mxa', 'mxb'
                ],
                'virus_specific_retrovirus': [
                    'hiv', 'lentivirus', 'retrovirus', 'hiv-1', 'hiv-2',
                    'murine leukemia virus', 'mlv', 'fiv', 'siv'
                ],
                'mechanism_retrovirus': [
                    'viral budding', 'virus release', 'vpu', 'vif', 'vpr', 'nef',
                    'capsid', 'nuclear import', 'uncoating', 'protease'
                ]
            }

            retrovirus_terms = ['hiv', 'lentivirus', 'retrovirus', 'mlv', 'apobec', 'trim5', 'tetherin']

            all_keywords = []
            for category, words in antiviral_keywords.items():
                all_keywords.extend(words)

            evidence = []
            for paper in papers:
                text = (paper.get('abstract', '') + ' ' + paper.get('title', '')).lower()

                has_retrovirus_context = any(term in text for term in retrovirus_terms)
                if not has_retrovirus_context:
                    continue

                match_score = 0
                matched_terms = []

                for term in all_keywords:
                    if term in text:
                        match_score += 1
                        matched_terms.append(term)

                title_lower = paper.get('title', '').lower()
                if any(term in title_lower for term in ['hiv', 'lentivirus', 'retrovirus', 'apobec', 'trim5']):
                    match_score += 3
                elif any(term in title_lower for term in ['antiviral', 'virus restriction']):
                    match_score += 1

                if match_score >= 4:
                    evidence.append({
                        **paper,
                        'match_score': match_score,
                        'matched_terms': matched_terms[:5],
                        'retrovirus_specific': True
                    })

            evidence.sort(key=lambda x: x.get('match_score', 0), reverse=True)
        else:
            target_phrases = {
                'essential': [
                    'lethal knockout', 'knockout is lethal', 'essential for survival',
                    'required for viability', 'knockout leads to death',
                    'deletion is lethal', 'null mutant lethal'
                ],
                'toxic': [
                    'overexpression induced cell death', 'overexpression is cytotoxic',
                    'overexpression triggers apoptosis', 'overexpression lethal',
                    'overexpression toxic', 'ectopic expression cell death'
                ]
            }.get(check_type, [])

            evidence = []
            for paper in papers:
                text = (paper.get('abstract', '') + ' ' + paper.get('title', '')).lower()
                if any(phrase in text for phrase in target_phrases):
                    evidence.append(paper)

        if evidence:
            pmid_list = [p['pmid'] for p in evidence[:3]]
            best_match = evidence[0]

            if check_type == 'antiviral' and 'match_score' in best_match:
                match_info = f"（匹配度: {best_match['match_score']}/12，关键词: {', '.join(best_match.get('matched_terms', [])[:3])}）"
            else:
                match_info = ""

            return HardRuleCheck(
                rule_name=f"{type_names[check_type]}检查（文献补充）",
                passed=False,
                reason=f"文献检索发现 {gene_name} 具有{type_names[check_type]}证据：{best_match['title'][:80]}...{match_info}",
                source=f"PubMed文献检索（{len(evidence)}篇明确证据，核心数据库未收录）",
                pmid=pmid_list[0],
                pmid_list=pmid_list,
                evidence_papers=evidence[:3],
                overrideable=False,
                check_level="literature"
            )

        return HardRuleCheck(
            rule_name=f"{type_names[check_type]}检查（文献补充）",
            passed=True,
            reason=f"检索到{len(papers)}篇文献，但未发现明确{type_names[check_type]}证据",
            source="PubMed文献检索（无明确证据）",
            pmid_list=[p['pmid'] for p in papers[:3]],
            evidence_papers=papers[:3],
            overrideable=True,
            check_level="literature"
        )

    def _check_by_literature_enhanced(self, gene_name: str, check_type: str) -> HardRuleCheck:
        if check_type != 'antiviral':
            return self._check_by_literature(gene_name, check_type)

        papers = self.ncbi.search_gene_property_literature(gene_name, check_type)

        if not papers:
            return HardRuleCheck(
                rule_name="抗病毒基因检查（文献+AI分析）",
                passed=True,
                reason=f"核心数据库未收录，且未检索到相关文献",
                source="核心数据库+PubMed（零结果）",
                overrideable=True,
                check_level="literature"
            )

        pre_filtered = []
        for paper in papers[:10]:
            text = (paper.get('abstract', '') + ' ' + paper.get('title', '')).lower()
            if any(kw in text for kw in ['virus', 'viral', 'infection', 'interferon', 'immunity', 'host']):
                pre_filtered.append(paper)

        if not pre_filtered:
            return HardRuleCheck(
                rule_name="抗病毒基因检查（文献+AI分析）",
                passed=True,
                reason=f"检索到{len(papers)}篇文献，但预筛选未发现有潜力文献",
                source="PubMed文献检索（关键词预筛选）",
                pmid_list=[p['pmid'] for p in papers[:3]],
                evidence_papers=papers[:3],
                overrideable=True,
                check_level="literature"
            )

        ai_evidence = []
        if self.ai and self.ai.api_key:
            with st.spinner(f"AI正在分析{len(pre_filtered)}篇文献的抗病毒证据..."):
                for paper in pre_filtered[:3]:
                    try:
                        analysis = self.ai.analyze_antiviral_evidence(
                            gene_name=gene_name,
                            title=paper.get('title', ''),
                            abstract=paper.get('abstract', '')
                        )
                        if analysis.get('is_antiviral') and analysis.get('confidence', 0) > 0.6:
                            ai_evidence.append({
                                **paper,
                                'ai_confidence': analysis.get('confidence'),
                                'ai_mechanism': analysis.get('mechanism'),
                                'ai_reasoning': analysis.get('reasoning')
                            })
                    except Exception as e:
                        logger.error(f"AI分析失败: {e}")
                        continue

        if ai_evidence:
            best = ai_evidence[0]
            pmid_list = [p['pmid'] for p in ai_evidence[:3]]
            mechanism = best.get('ai_mechanism', '未知机制')
            confidence = best.get('ai_confidence', 0)

            return HardRuleCheck(
                rule_name="抗病毒基因检查（文献+AI分析）",
                passed=False,
                reason=f"AI分析确认 {gene_name} 具有抗病毒功能（置信度: {confidence:.0%}）",
                source=f"PubMed + 通义千问AI语义分析（机制: {mechanism}）",
                pmid=best['pmid'],
                pmid_list=pmid_list,
                evidence_papers=ai_evidence[:3],
                overrideable=False,
                check_level="literature"
            )

        return self._check_by_literature(gene_name, check_type)

# ==================== 多数据库转录本选择引擎 ====================
class TranscriptSelector:
    def __init__(self, ncbi_client: NCBIClient, email: str):
        self.ncbi = ncbi_client
        self.email = email
        self.cache = {}

    def select_optimal_transcript(self, gene_name: str, gene_id: str,
                                 cell_line: Optional[str] = None) -> Dict:
        results = {
            'gene': gene_name,
            'gene_id': gene_id,
            'selected_transcript': None,
            'selection_reason': [],
            'all_transcripts': [],
            'database_coverage': {},
            'database_errors': {},  # 新增：记录每个数据库的错误
            'conflicts': [],
            'filtered_xm': []
        }

        with st.spinner("多数据库交叉验证转录本（APPRIS/NCBI/Ensembl，优先NM）..."):
            # 获取转录本数据并记录错误
            ncbi_data, ncbi_error = self._fetch_ncbi_refseq_with_error(gene_id)
            results['database_coverage']['NCBI_RefSeq'] = len(ncbi_data)
            if ncbi_error:
                results['database_errors']['NCBI_RefSeq'] = ncbi_error

            ensembl_data, ensembl_error = self._fetch_ensembl_with_error(gene_name)
            results['database_coverage']['Ensembl'] = len(ensembl_data)
            if ensembl_error:
                results['database_errors']['Ensembl'] = ensembl_error

            appris_data, appris_error = self._fetch_appris_with_error(gene_name)
            results['database_coverage']['APPRIS'] = len(appris_data)
            if appris_error:
                results['database_errors']['APPRIS'] = appris_error

            ccle_data = {}
            if cell_line:
                ccle_data, ccle_error = self._fetch_ccle_with_error(gene_name, cell_line)
                results['database_coverage']['CCLE'] = len(ccle_data)
                if ccle_error:
                    results['database_errors']['CCLE'] = ccle_error

            # 详细调试信息显示
            with st.expander("🔍 转录本获取诊断信息", expanded=True):  # 默认展开
                st.write(f"**查询参数**:")
                st.write(f"- 基因名称: `{gene_name}`")
                st.write(f"- NCBI Gene ID: `{gene_id}`")
                
                # 显示各数据库状态
                st.write("**数据库查询状态**:")
                
                # NCBI
                if 'NCBI_RefSeq' in results['database_errors']:
                    st.error(f"❌ NCBI RefSeq: {results['database_errors']['NCBI_RefSeq']}")
                else:
                    st.success(f"✓ NCBI RefSeq: 找到 {len(ncbi_data)} 个转录本")
                    if ncbi_data:
                        st.write(f"  - 转录本列表: {', '.join(list(ncbi_data.keys())[:5])}")
                
                # Ensembl
                if 'Ensembl' in results['database_errors']:
                    st.error(f"❌ Ensembl: {results['database_errors']['Ensembl']}")
                else:
                    st.success(f"✓ Ensembl: 找到 {len(ensembl_data)} 个转录本")
                
                # APPRIS
                if 'APPRIS' in results['database_errors']:
                    st.error(f"❌ APPRIS: {results['database_errors']['APPRIS']}")
                else:
                    st.success(f"✓ APPRIS: 找到 {len(appris_data)} 个转录本")
                
                # CCLE
                if cell_line:
                    if 'CCLE' in results['database_errors']:
                        st.error(f"❌ CCLE: {results['database_errors']['CCLE']}")
                    else:
                        st.success(f"✓ CCLE: 找到 {len(ccle_data)} 个表达记录")
                
                # 显示网络诊断建议
                if len(results['database_errors']) >= 3:
                    st.warning("⚠️ **诊断建议**:")
                    st.markdown("""
                    1. **检查网络连接**: Streamlit Cloud 可能无法访问 NCBI/Ensembl API
                    2. **配置 NCBI API Key**: 在侧边栏输入有效的 NCBI API Key 可提高成功率
                    3. **检查基因ID**: 确认 `{}` 是有效的 NCBI Gene ID
                    4. **备选方案**: 如果持续失败，建议手动输入转录本信息
                    """.format(gene_id))

            all_transcripts = self._merge_transcript_sources(
                ncbi_data, ensembl_data, appris_data, ccle_data
            )

            if not all_transcripts:
                st.warning("⚠️ 所有数据库均未返回有效转录本")
                # 尝试备选方案
                fallback_tx = self._create_fallback_transcript(gene_name, gene_id, diagnosis=results['database_errors'])
                if fallback_tx:
                    fallback_tx['diagnosis'] = results['database_errors']
                    return fallback_tx
                return {'error': '未找到任何转录本信息', 'fallback': True, 'diagnosis': results['database_errors']}

            # 过滤XM转录本，优先NM
            nm_transcripts = {}
            xm_transcripts = {}

            for tx_id, tx_info in all_transcripts.items():
                if tx_id.startswith('NM_'):
                    nm_transcripts[tx_id] = tx_info
                elif tx_id.startswith('XM_'):
                    xm_transcripts[tx_id] = tx_info
                else:
                    nm_transcripts[tx_id] = tx_info

            results['filtered_xm'] = list(xm_transcripts.keys())

            transcripts_to_score = nm_transcripts if nm_transcripts else xm_transcripts

            if not transcripts_to_score:
                return {'error': '未找到有效转录本（NM或XM）', 'fallback': True}

            scored_transcripts = []
            for tx_id, tx_info in transcripts_to_score.items():
                score, reasons = self._calculate_transcript_score(tx_info, cell_line)
                if tx_id.startswith('NM_'):
                    score += 0.3
                    reasons.append("NCBI RefSeq NM（已验证转录本，优先）")
                elif tx_id.startswith('XM_'):
                    reasons.append("NCBI RefSeq XM（预测转录本，次选）")

                scored_transcripts.append({
                    'id': tx_id,
                    'score': score,
                    'info': tx_info,
                    'reasons': reasons
                })

            scored_transcripts.sort(key=lambda x: x['score'], reverse=True)
            results['all_transcripts'] = scored_transcripts

            best = scored_transcripts[0]
            results['selected_transcript'] = best

            if len(scored_transcripts) > 1:
                second_best = scored_transcripts[1]
                if best['score'] - second_best['score'] < 0.2:
                    results['conflicts'] = [best, second_best]
                    results['needs_confirmation'] = True

        return results

    def _fetch_ncbi_refseq_with_error(self, gene_id: str) -> Tuple[Dict, Optional[str]]:
        """获取NCBI RefSeq转录本，返回数据和错误信息"""
        try:
            data = self._fetch_ncbi_refseq(gene_id)
            return data, None
        except Exception as e:
            error_msg = str(e)
            logger.error(f"NCBI RefSeq获取失败: {error_msg}")
            return {}, f"API请求失败: {error_msg[:100]}"

    def _fetch_ensembl_with_error(self, gene_name: str) -> Tuple[Dict, Optional[str]]:
        """获取Ensembl转录本，返回数据和错误信息"""
        try:
            data = self._fetch_ensembl(gene_name)
            return data, None
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Ensembl获取失败: {error_msg}")
            return {}, f"API请求失败: {error_msg[:100]}"

    def _fetch_appris_with_error(self, gene_name: str) -> Tuple[Dict, Optional[str]]:
        """获取APPRIS转录本，返回数据和错误信息"""
        try:
            data = self._fetch_appris(gene_name)
            return data, None
        except Exception as e:
            error_msg = str(e)
            logger.error(f"APPRIS获取失败: {error_msg}")
            return {}, f"API请求失败: {error_msg[:100]}"

    def _fetch_ccle_with_error(self, gene_name: str, cell_line: str) -> Tuple[Dict, Optional[str]]:
        """获取CCLE表达数据，返回数据和错误信息"""
        try:
            data = self._fetch_ccle_expression(gene_name, cell_line)
            return data, None
        except Exception as e:
            error_msg = str(e)
            logger.error(f"CCLE获取失败: {error_msg}")
            return {}, f"数据查询失败: {error_msg[:100]}"

    def _fetch_ncbi_refseq(self, gene_id: str) -> Dict:
        """使用NCBI E-utilities获取RefSeq转录本（使用改进的NCBI方法）"""
        transcripts = {}
        try:
            # 使用NCBIClient的_fetch_transcripts方法来获取（已带重试）
            tx_list = self.ncbi._fetch_transcripts(gene_id)
            
            if not tx_list:
                logger.warning(f"NCBI未返回基因ID {gene_id} 的任何转录本")
                # 检查NCBI客户端配置
                if not self.ncbi.email or '@' not in self.ncbi.email:
                    st.warning("⚠️ NCBI Email 配置无效，请在侧边栏配置有效的邮箱地址")
                elif not self.ncbi.api_key:
                    st.info("💡 提示：配置 NCBI API Key 可提高请求成功率和限额")
            else:
                logger.info(f"NCBI返回 {len(tx_list)} 个转录本")
            
            for tx in tx_list:
                # 防御性检查：跳过 None 值
                if not tx:
                    continue
                tx_id = tx.get('id')
                if not tx_id:
                    continue
                transcripts[tx_id] = {
                    'source': 'NCBI',
                    'status': tx.get('status', ''),
                    'length': tx.get('length', 0),
                    'type': tx.get('type', ''),
                    'title': tx.get('title', '')
                }
            
            if transcripts:
                logger.info(f"NCBI RefSeq获取成功: {len(transcripts)}个转录本")
            
            # 调试信息已移除
            
        except Exception as e:
            logger.error(f"NCBI RefSeq获取失败: {e}")
            st.warning(f"⚠️ NCBI RefSeq获取失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
        return transcripts

    def _fetch_ensembl(self, gene_name: str) -> Dict:
        transcripts = {}
        try:
            server = "https://rest.ensembl.org"
            ext = f"/lookup/symbol/homo_sapiens/{gene_name}?expand=1"

            response = requests.get(server + ext,
                                  headers={"Content-Type": "application/json"},
                                  timeout=10)

            if response.status_code == 200:
                data = response.json()
                for transcript in data.get('Transcript', []):
                    tx_id = transcript.get('id')
                    # 防御性检查：跳过无效ID
                    if not tx_id:
                        continue
                    appris = transcript.get('appris', '')
                    tsl = transcript.get('tsl', '')
                    biotype = transcript.get('biotype', '')

                    translation = transcript.get('Translation', {})
                    cds_length = translation.get('length', 0) * 3 if translation else 0

                    transcripts[tx_id] = {
                        'source': 'Ensembl',
                        'appris_tag': appris,
                        'tsl': tsl,
                        'biotype': biotype,
                        'length': cds_length
                    }
        except Exception as e:
            logger.error(f"Ensembl获取失败: {e}")
        return transcripts

    def _fetch_appris(self, gene_name: str) -> Dict:
        transcripts = {}
        try:
            url = f"https://apprisws.bioinfo.cnio.es/rest/v1.22/homo_sapiens/e/{gene_name}"
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                data = response.json()
                for tx in data.get('transcripts', []):
                    tx_id = tx.get('id')
                    # 防御性检查：跳过无效ID
                    if not tx_id:
                        continue
                    label = tx.get('principal_isoform', '')
                    firestar = tx.get('firestar_score', 0)

                    transcripts[tx_id] = {
                        'source': 'APPRIS',
                        'principal_label': label,
                        'firestar_score': firestar,
                        'reliability': 'HIGH' if label == 'PRINCIPAL' else 'MEDIUM'
                    }
        except Exception as e:
            logger.error(f"APPRIS获取失败: {e}")
        return transcripts

    def _fetch_ccle_expression(self, gene_name: str, cell_line: str) -> Dict:
        expression_data = {}
        try:
            ccle_name = cell_line.replace('-', '').upper()
            logger.info(f"查询CCLE表达: {gene_name} in {ccle_name}")
        except Exception as e:
            logger.warning(f"CCLE查询失败: {e}")
        return expression_data

    def _merge_transcript_sources(self, ncbi: Dict, ensembl: Dict,
                                 appris: Dict, ccle: Dict) -> Dict:
        merged = {}

        for refseq_id, ncbi_info in ncbi.items():
            # 防御性检查：跳过 None 值
            if not ncbi_info:
                continue
                
            ensembl_id = None

            merged[refseq_id] = {
                'refseq_id': refseq_id,
                'ensembl_id': ensembl_id,
                'ncbi_status': ncbi_info.get('status'),
                'length': ncbi_info.get('length', 0),
                'appris_label': appris.get(ensembl_id, {}).get('principal_label', 'UNKNOWN'),
                'ccle_tpm': ccle.get(ensembl_id, {}).get('tpm', 0),
                'sources': ['NCBI']
            }

            for ens_id, ens_info in ensembl.items():
                if refseq_id in str(ens_info):
                    merged[refseq_id]['sources'].append('Ensembl')
                    merged[refseq_id]['tsl'] = ens_info.get('tsl')
                    merged[refseq_id]['biotype'] = ens_info.get('biotype')
                    if merged[refseq_id]['length'] == 0 and ens_info.get('length', 0) > 0:
                        merged[refseq_id]['length'] = ens_info['length']

        return merged

    def _create_fallback_transcript(self, gene_name: str, gene_id: str, **kwargs) -> Optional[Dict]:
        """当所有数据库都失败时，尝试通过NCBI Datasets网页爬取MANE Select转录本"""
        
        # 尝试通过网页爬取获取MANE Select转录本
        mane_transcript = self._fetch_mane_select_from_web(gene_id)
        
        if mane_transcript:
            return {
                'gene': gene_name,
                'gene_id': gene_id,
                'selected_transcript': {
                    'id': mane_transcript['id'],
                    'score': 0.5,
                    'info': {
                        'source': 'NCBI_Datasets_Web',
                        'status': 'MANE_Select',
                        'length': mane_transcript['length'],
                        'type': 'NM',
                        'title': f"{mane_transcript['id']} (MANE Select)"
                    },
                    'reasons': [f"MANE Select转录本（网页爬取）", f"长度: {mane_transcript['length']}bp"]
                },
                'all_transcripts': [{
                    'id': mane_transcript['id'],
                    'score': 0.5,
                    'info': {
                        'source': 'NCBI_Datasets_Web',
                        'status': 'MANE_Select',
                        'length': mane_transcript['length'],
                        'type': 'NM',
                        'title': f"{mane_transcript['id']} (MANE Select)"
                    },
                    'reasons': [f"MANE Select转录本（网页爬取）", f"长度: {mane_transcript['length']}bp"]
                }],
                'database_coverage': {'NCBI_RefSeq': 0, 'Ensembl': 0, 'APPRIS': 0, 'NCBI_Datasets_Web': 1},
                'conflicts': [],
                'note': '通过NCBI Datasets网页爬取获得MANE Select转录本'
            }
        
        # 如果网页爬取也失败，返回错误状态
        diagnosis = kwargs.get('diagnosis', {})
        return {
            'gene': gene_name,
            'gene_id': gene_id,
            'selected_transcript': None,
            'all_transcripts': [],
            'database_coverage': {'NCBI_RefSeq': 0, 'Ensembl': 0, 'APPRIS': 0},
            'conflicts': [],
            'error': 'NCBI/Ensembl/APPRIS数据库均未返回有效转录本，且网页爬取失败',
            'note': '请检查：1)NCBI API配置是否正确 2)基因名称是否正确 3)网络连接状态 4)Streamlit Cloud网络限制',
            'diagnosis': diagnosis
        }
    
    def _fetch_mane_select_from_web(self, gene_id: str) -> Optional[Dict]:
        """通过NCBI Datasets网页爬取MANE Select转录本信息
        
        访问: https://www.ncbi.nlm.nih.gov/datasets/gene/gene{id}/#transcripts-and-proteins
        提取标注为"MANE Select"的转录本的Length(nt)
        """
        import requests
        import re
        
        # 尝试导入 BeautifulSoup，如果失败则使用正则备选方案
        try:
            from bs4 import BeautifulSoup
            HAS_BS4 = True
        except ImportError:
            HAS_BS4 = False
            logger.warning("BeautifulSoup4 未安装，将使用正则表达式备选方案")
        
        if not gene_id or not gene_id.isdigit():
            logger.warning(f"无效的gene_id: {gene_id}")
            return None
        
        url = f"https://www.ncbi.nlm.nih.gov/datasets/gene/{gene_id}/"
        
        try:
            logger.info(f"尝试从NCBI Datasets网页获取MANE Select: {url}")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
            }
            
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            html_content = response.text
            
            # 如果有 BeautifulSoup，使用它解析
            if HAS_BS4:
                return self._parse_mane_with_bs4(html_content, gene_id)
            else:
                # 使用正则表达式备选方案
                return self._parse_mane_with_regex(html_content, gene_id)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"请求NCBI Datasets网页失败: {e}")
            return None
        except Exception as e:
            logger.error(f"解析NCBI Datasets网页失败: {e}")
            return None
    
    def _parse_mane_with_bs4(self, html_content: str, gene_id: str) -> Optional[Dict]:
        """使用BeautifulSoup解析MANE Select信息"""
        from bs4 import BeautifulSoup
        import re
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # 方法1: 查找所有包含"MANE Select"文本的元素
        for element in soup.find_all(text=re.compile('MANE Select', re.IGNORECASE)):
            parent = element.parent
            
            # 向上查找包含转录本信息的父元素
            for _ in range(5):  # 最多向上查找5层
                if parent:
                    # 查找转录本ID (NM_或XM_开头)
                    parent_text = parent.get_text()
                    
                    # 查找转录本ID
                    tx_match = re.search(r'(NM_\d+(\.\d+)?)', parent_text)
                    if not tx_match:
                        tx_match = re.search(r'(XM_\d+(\.\d+)?)', parent_text)
                    
                    if tx_match:
                        transcript_id = tx_match.group(1)
                        
                        # 查找长度信息 - 多种可能的格式
                        length = None
                        
                        # 尝试匹配 "Length: 1234 nt" 或 "Length(nt): 1234"
                        length_match = re.search(r'Length[:\s\(]*(?:nt)?[:\s\)]*(\d+)', parent_text, re.IGNORECASE)
                        if length_match:
                            length = int(length_match.group(1))
                        
                        # 尝试匹配 "1234 bp" 或 "1234 nt"
                        if not length:
                            length_match = re.search(r'(\d+)\s*(?:bp|nt|nucleotides?)', parent_text, re.IGNORECASE)
                            if length_match:
                                length = int(length_match.group(1))
                        
                        # 如果在当前元素没找到长度，尝试在兄弟元素或表格中查找
                        if not length:
                            # 查找表格行
                            row = parent.find_parent('tr') if hasattr(parent, 'find_parent') else None
                            if row:
                                row_text = row.get_text()
                                length_match = re.search(r'(\d+)', row_text.replace(',', ''))
                                if length_match:
                                    potential_length = int(length_match.group(1))
                                    if 100 < potential_length < 50000:  # 合理的转录本长度范围
                                        length = potential_length
                        
                        if transcript_id and length:
                            logger.info(f"找到MANE Select转录本: {transcript_id}, 长度: {length}bp")
                            return {
                                'id': transcript_id,
                                'length': length,
                                'source': 'MANE_Select'
                            }
                    
                    parent = parent.parent
        
        # 方法2: 查找所有表格，寻找包含MANE Select信息的行
        tables = soup.find_all('table')
        for table in tables:
            rows = table.find_all('tr')
            for row in rows:
                row_text = row.get_text()
                if 'MANE Select' in row_text or 'MANE' in row_text:
                    # 查找转录本ID
                    tx_match = re.search(r'(NM_\d+(\.\d+)?)', row_text)
                    if not tx_match:
                        tx_match = re.search(r'(XM_\d+(\.\d+)?)', row_text)
                    
                    if tx_match:
                        transcript_id = tx_match.group(1)
                        
                        # 尝试从行中提取长度
                        cells = row.find_all(['td', 'th'])
                        for cell in cells:
                            cell_text = cell.get_text().replace(',', '')
                            # 查找数字
                            num_match = re.search(r'^(\d+)$', cell_text.strip())
                            if num_match:
                                potential_length = int(num_match.group(1))
                                if 100 < potential_length < 50000:
                                    logger.info(f"从表格找到MANE Select转录本: {transcript_id}, 长度: {potential_length}bp")
                                    return {
                                        'id': transcript_id,
                                        'length': potential_length,
                                        'source': 'MANE_Select'
                                    }
        
        logger.warning(f"未在NCBI Datasets网页找到MANE Select转录本 (gene_id: {gene_id})")
        return None
    
    def _parse_mane_with_regex(self, html_content: str, gene_id: str) -> Optional[Dict]:
        """使用正则表达式解析MANE Select信息（备选方案）"""
        import re
        
        # 清理HTML标签以便更好地搜索
        text_content = re.sub(r'<script[^>]*>.*?</script>', ' ', html_content, flags=re.DOTALL)
        text_content = re.sub(r'<style[^>]*>.*?</style>', ' ', text_content, flags=re.DOTALL)
        text_content = re.sub(r'<[^>]+>', ' ', text_content)
        text_content = re.sub(r'\s+', ' ', text_content)
        
        # 查找包含MANE Select的区域
        # 尝试匹配 "MANE Select" 附近的转录本信息
        mane_patterns = [
            # 模式1: MANE Select 后跟转录本ID和长度
            r'MANE Select.*?((?:NM|XM)_\d+(?:\.\d+)?).*?(\d{3,5})\s*(?:bp|nt|nucleotides?)',
            # 模式2: 转录本ID后跟MANE Select，然后是长度
            r'((?:NM|XM)_\d+(?:\.\d+)?).*?MANE Select.*?(\d{3,5})\s*(?:bp|nt|nucleotides?)',
            # 模式3: 表格格式，查找有MANE字样的行
            r'MANE[^\n]*?((?:NM|XM)_\d+(?:\.\d+)?)[^\n]*?(\d{3,5})',
        ]
        
        for pattern in mane_patterns:
            matches = re.findall(pattern, text_content, re.IGNORECASE | re.DOTALL)
            for match in matches:
                transcript_id = match[0]
                try:
                    length = int(match[1])
                    if 100 < length < 50000:
                        logger.info(f"[正则] 找到MANE Select转录本: {transcript_id}, 长度: {length}bp")
                        return {
                            'id': transcript_id,
                            'length': length,
                            'source': 'MANE_Select'
                        }
                except (ValueError, IndexError):
                    continue
        
        # 如果上面的模式没找到，尝试更宽松的匹配
        # 先找MANE Select，然后在其附近找NM_开头的ID和数字
        mane_positions = [m.start() for m in re.finditer(r'MANE', text_content, re.IGNORECASE)]
        
        for pos in mane_positions:
            # 提取MANE附近的一段文本
            start = max(0, pos - 500)
            end = min(len(text_content), pos + 500)
            nearby_text = text_content[start:end]
            
            # 查找转录本ID
            tx_match = re.search(r'(NM_\d+(?:\.\d+)?)', nearby_text)
            if not tx_match:
                tx_match = re.search(r'(XM_\d+(?:\.\d+)?)', nearby_text)
            
            if tx_match:
                transcript_id = tx_match.group(1)
                
                # 在附近查找长度（通常是3-5位的数字）
                length_matches = re.findall(r'\b(\d{3,5})\b', nearby_text)
                for lm in length_matches:
                    length = int(lm)
                    if 300 < length < 15000:  # 更严格的合理范围
                        logger.info(f"[正则备选] 找到MANE Select转录本: {transcript_id}, 长度: {length}bp")
                        return {
                            'id': transcript_id,
                            'length': length,
                            'source': 'MANE_Select'
                        }
        
        logger.warning(f"[正则] 未找到MANE Select转录本 (gene_id: {gene_id})")
        return None

    def _calculate_transcript_score(self, tx_info: Dict, cell_line: Optional[str]) -> Tuple[float, List[str]]:
        # 防御性检查：如果 tx_info 为 None，返回默认分数
        if not tx_info:
            return 0.0, ["转录本信息不可用"]
        
        score = 0.0
        reasons = []

        appris = tx_info.get('appris_label', '')
        if appris == 'PRINCIPAL':
            score += 0.4
            reasons.append("APPRIS Principal Isoform（主要转录本）")
        elif 'ALTERNATIVE' in appris:
            score += 0.2
            reasons.append("APPRIS Alternative Isoform")

        status = tx_info.get('ncbi_status', '')
        if status == 'REVIEWED':
            score += 0.3
            reasons.append("NCBI RefSeq REVIEWED（专家审核）")
        elif status == 'VALIDATED':
            score += 0.2
            reasons.append("NCBI RefSeq VALIDATED")
        elif status == 'PROVISIONAL':
            score += 0.1
            reasons.append("NCBI RefSeq PROVISIONAL")

        tpm = tx_info.get('ccle_tpm', 0)
        if cell_line and tpm > 0:
            if tpm > 10:
                score += 0.2
                reasons.append(f"在{cell_line}中高表达（TPM={tpm:.1f}）")
            elif tpm > 1:
                score += 0.1
                reasons.append(f"在{cell_line}中低表达（TPM={tpm:.1f}）")

        if tx_info.get('biotype') == 'nonsense_mediated_decay':
            score -= 0.5
            reasons.append("NMD降解转录本（含提前终止密码子）")
        elif tx_info.get('biotype') == 'protein_coding':
            score += 0.1
            reasons.append("蛋白编码转录本")

        length = tx_info.get('length', 0)
        if length < 300 and length > 0:
            score -= 0.2
            reasons.append(f"较短CDS（{length}bp，可能为截断体）")
        elif length >= 300:
            reasons.append(f"CDS长度 {length}bp")

        return max(0, score), reasons

# ==================== 基因输入组件 ====================
class GeneAutocompleteService:
    def __init__(self):
        self.clinical_tables_url = "https://clinicaltables.nlm.nih.gov/api/ncbi_genes/v3/search"

    @safe_cache_data
    def get_suggestions(_self, query: str, organism: str = "human", limit: int = 8) -> List[Dict]:
        if not query or len(query) < 2:
            return []

        try:
            organism_map = {
                "human": "Homo sapiens",
                "mouse": "Mus musculus",
                "rat": "Rattus norvegicus",
                "cho": "Cricetulus griseus",
                "pig": "Sus scrofa",
                "monkey": "Macaca mulatta"
            }
            organism_name = organism_map.get(organism, organism)

            params = {
                "terms": query,
                "maxList": limit,
                "df": "symbol,name,chromosome,gene_id,type_of_gene",
                "q": f"organism:\"{organism_name}\""
            }

            response = requests.get(_self.clinical_tables_url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()

            if data and len(data) >= 3:
                results = []
                headers = data[0]
                rows = data[2]

                for row in rows:
                    gene_info = dict(zip(headers, row))
                    results.append({
                        "symbol": html.escape(gene_info.get("symbol", "")),
                        "name": html.escape(gene_info.get("name", "")),
                        "gene_id": gene_info.get("gene_id", ""),
                        "chromosome": html.escape(gene_info.get("chromosome", "")),
                        "type": html.escape(gene_info.get("type_of_gene", ""))
                    })
                return results
            return []

        except Exception as e:
            logger.warning(f"Gene suggestion error: {e}")
            return []

class GeneInputComponent:
    """基因输入组件 - 完全基于HPA数据包（抛弃NCBI自动补全）"""
    
    def __init__(self, hpa_gene_service: HPAGeneAutocompleteService = None, hpa_detail_service: HPAGeneDetailService = None):
        self.hpa_gene_service = hpa_gene_service
        self.hpa_detail_service = hpa_detail_service

    def render(self, organism: str = "human", key_prefix: str = "gene", disabled: bool = False) -> Optional[str]:
        input_key = f"{key_prefix}_input"
        selected_key = f"{key_prefix}_selected"
        suggestions_key = f"{key_prefix}_suggestions"
        last_query_key = f"{key_prefix}_last_query"
        hpa_info_key = f"{key_prefix}_hpa_info"

        if input_key not in st.session_state:
            st.session_state[input_key] = ""
        if selected_key not in st.session_state:
            st.session_state[selected_key] = ""
        if suggestions_key not in st.session_state:
            st.session_state[suggestions_key] = []
        if last_query_key not in st.session_state:
            st.session_state[last_query_key] = ""
        if hpa_info_key not in st.session_state:
            st.session_state[hpa_info_key] = None

        # 统一使用HPA数据包进行基因自动补全
        hpa_available = (self.hpa_gene_service is not None)
        
        # 检查HPA数据文件是否存在
        hpa_data_ready = False
        if hpa_available and self.hpa_gene_service:
            # 检查搜索索引是否已构建（即数据文件是否存在）
            if hasattr(self.hpa_gene_service, 'search_index') and self.hpa_gene_service.search_index:
                hpa_data_ready = True
        
        input_label = "基因名（HPA数据自动补全，输入2个字符以上显示建议）"
        
        # 如果HPA数据未就绪，显示警告并尝试下载
        if hpa_available and not hpa_data_ready:
            st.warning("⚠️ HPA基因数据尚未下载。首次使用需要下载约200MB数据，请稍候...")
            # 尝试触发下载
            try:
                hpa_manager = getattr(self.hpa_gene_service, 'hpa_manager', None)
                if hpa_manager and hasattr(hpa_manager, 'check_and_download'):
                    with st.spinner("正在下载HPA数据（约200MB）..."):
                        hpa_manager.check_and_download()
                    # 下载完成后重建索引
                    if hasattr(self.hpa_gene_service, 'rebuild_index'):
                        with st.spinner("正在构建基因索引..."):
                            success = self.hpa_gene_service.rebuild_index()
                            if success:
                                st.success(f"✅ 索引构建完成！共 {len(self.hpa_gene_service.search_index)} 个基因名称")
                                hpa_data_ready = True
                            else:
                                st.error("❌ 索引构建失败")
            except Exception as e:
                logger.error(f"HPA数据下载失败: {e}")
                st.error(f"数据下载失败: {e}")

        user_input = st.text_input(
            input_label,
            value=st.session_state[input_key],
            placeholder="例如：TP53, EGFR, GAPDH...",
            key=f"{key_prefix}_text_widget",
            disabled=disabled
        )

        if user_input != st.session_state[input_key]:
            st.session_state[input_key] = user_input
            if st.session_state[selected_key] and user_input != st.session_state[selected_key]:
                st.session_state[selected_key] = ""
                st.session_state[suggestions_key] = []
                st.session_state[hpa_info_key] = None

            if len(user_input) >= 2:
                safe_rerun()

        if len(user_input) >= 2 and not st.session_state[selected_key]:
            last_query = st.session_state.get(last_query_key, "")
            if user_input != last_query:
                if hpa_available and hpa_data_ready:
                    # 使用HPA自动补全（基于Gene synonym列）
                    hpa_suggestions = self.hpa_gene_service.get_suggestions(user_input, limit=8)
                    suggestions = []
                    for sug in hpa_suggestions:
                        gene_info = self.hpa_gene_service.get_gene_info(sug['gene_symbol'])
                        suggestions.append({
                            "symbol": sug['gene_symbol'],
                            "name": gene_info.get('description', '') if gene_info else '',
                            "gene_id": gene_info.get('ensembl_id', '') if gene_info else '',
                            "chromosome": gene_info.get('chromosome', '') if gene_info else '',
                            "type": "protein-coding",
                            "source": "HPA",
                            "match_type": sug.get('match_type', ''),
                            "matched_name": sug.get('matched_name', ''),
                            "name_type": sug.get('name_type', '')  # primary或synonym
                        })
                else:
                    # HPA服务不可用时的降级处理
                    suggestions = []
                
                st.session_state[suggestions_key] = suggestions
                st.session_state[last_query_key] = user_input
                safe_rerun()

        suggestions = st.session_state.get(suggestions_key, [])
        if suggestions and not st.session_state[selected_key]:
            st.caption(f"💡 HPA基因匹配 ({len(suggestions)}个建议)：")
            cols = st.columns(min(len(suggestions), 4))
            for i, gene in enumerate(suggestions):
                with cols[i % 4]:
                    display_text = f"{gene['symbol']}"
                    match_type = gene.get('match_type', '')
                    matched_name = gene.get('matched_name', '')
                    name_type = gene.get('name_type', '')
                    
                    # 显示匹配类型图标
                    icons = {'exact': '✓', 'prefix': '↳', 'substring': '~', 'fuzzy': '≈'}
                    icon = icons.get(match_type, '•')
                    
                    # 如果匹配的是synonym，显示原始匹配名
                    if name_type == 'synonym' and matched_name and matched_name.upper() != gene['symbol'].upper():
                        display_text = f"{icon} {gene['symbol']} (via {matched_name})"
                    else:
                        display_text = f"{icon} {gene['symbol']}"
                    
                    btn_type = "primary" if match_type == 'exact' else "secondary"
                    help_text = f"匹配: {matched_name} ({match_type})" if matched_name else f"匹配类型: {match_type}"
                    
                    if st.button(display_text, key=f"{key_prefix}_sug_{i}", use_container_width=True, type=btn_type, help=help_text):
                        st.session_state[selected_key] = gene['symbol']
                        st.session_state[input_key] = gene['symbol']
                        st.session_state[f"{key_prefix}_info"] = gene
                        
                        # 获取HPA详细信息
                        if self.hpa_detail_service:
                            hpa_details = self.hpa_detail_service.get_gene_details(gene['symbol'])
                            st.session_state[hpa_info_key] = hpa_details
                        
                        safe_rerun()

        if st.session_state[selected_key]:
            gene_symbol = st.session_state[selected_key]
            
            # 如果还没有HPA详情，获取它
            if self.hpa_detail_service and not st.session_state.get(hpa_info_key):
                hpa_details = self.hpa_detail_service.get_gene_details(gene_symbol)
                st.session_state[hpa_info_key] = hpa_details
            
            # 显示选择信息
            if f"{key_prefix}_info" in st.session_state:
                gene_info = st.session_state[f"{key_prefix}_info"]
                match_info = ""
                if gene_info.get('name_type') == 'synonym' and gene_info.get('matched_name'):
                    match_info = f" (匹配自: {gene_info['matched_name']})"
                st.success(f"✓ 已选择HPA基因: **{gene_info['symbol']}**{match_info}")
            
            # 显示HPA详细信息面板
            if st.session_state.get(hpa_info_key):
                self._render_hpa_gene_info(st.session_state[hpa_info_key])
            
            return gene_symbol
        elif user_input:
            return user_input

        return None
    
    def _render_hpa_gene_info(self, gene_info):
        """渲染HPA基因详细信息面板 - 完整展示所有提取信息"""
        if not gene_info:
            return
        
        with st.expander("🔬 HPA基因详细信息", expanded=True):
            # ===== 第一行：基础信息 =====
            st.markdown("#### 📋 基础信息")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # a. Ensembl ID 链接
                if gene_info.get('ensembl_id'):
                    st.markdown(f"**Ensembl ID:**  ")
                    st.markdown(f"[{gene_info['ensembl_id']}]({gene_info['ensembl_url']})")
                else:
                    st.markdown("**Ensembl ID:** -")
            
            with col2:
                # b. Uniprot ID 链接
                if gene_info.get('uniprot_id'):
                    st.markdown(f"**Uniprot ID:**  ")
                    st.markdown(f"[{gene_info['uniprot_id']}]({gene_info['uniprot_url']})")
                else:
                    st.markdown("**Uniprot ID:** -")
            
            with col3:
                # c. 基因组位置（UCSC格式）
                if gene_info.get('genome_location'):
                    st.markdown(f"**基因组位置:**  ")
                    st.markdown(f"`{gene_info['genome_location']}`")
                    # 添加UCSC链接
                    ucsc_url = f"https://genome.ucsc.edu/cgi-bin/hgGene?hgg_gene={gene_info['gene_symbol']}"
                    st.caption(f"[🔗 UCSC Genome Browser]({ucsc_url})")
                else:
                    st.markdown("**基因组位置:** -")
            
            st.divider()
            
            # ===== 第二行：蛋白定位与功能 =====
            st.markdown("#### 🎯 蛋白定位与功能")
            loc = gene_info.get('protein_localization', {})
            func = gene_info.get('protein_function', {})
            
            loc_col1, loc_col2 = st.columns(2)
            with loc_col1:
                st.markdown("**亚细胞定位**")
                if loc.get('subcellular_main'):
                    st.markdown(f"- **主要定位:** {loc['subcellular_main']}")
                if loc.get('subcellular_additional'):
                    st.markdown(f"- **其他定位:** {loc['subcellular_additional']}")
                if loc.get('secretome_location'):
                    st.markdown(f"- **分泌组定位:** {loc['secretome_location']}")
                if loc.get('secretome_function'):
                    st.markdown(f"- **分泌组功能:** {loc['secretome_function']}")
            
            with loc_col2:
                st.markdown("**功能与疾病**")
                if func.get('biological_process'):
                    st.markdown(f"- **生物学过程:** {func['biological_process']}")
                if func.get('molecular_function'):
                    st.markdown(f"- **分子功能:** {func['molecular_function']}")
                if func.get('disease_involvement'):
                    st.markdown(f"- **疾病关联:** {func['disease_involvement']}")
            
            st.divider()
            
            # ===== 第三行：RNA表达数据（整合为单一表格）=====
            st.markdown("#### 📊 RNA表达数据")
            
            # 收集所有RNA表达数据
            rna_exp = gene_info.get('rna_expression', {})
            dist = gene_info.get('rna_distribution', {})
            
            # 构建RNA表达表格数据
            rna_table_data = []
            
            # 组织数据
            tissue = dist.get('tissue', {})
            tissue_specificity = tissue.get('specificity') or rna_exp.get('tissue_specificity', '')
            tissue_ntpm = tissue.get('specific_ntpm') or rna_exp.get('tissue_specific_ntpm', '')
            if tissue_specificity:
                rna_table_data.append({
                    "样本类型": "🧬 组织",
                    "表达特异性": tissue_specificity,
                    "表达量": f"nTPM: {tissue_ntpm}" if tissue_ntpm else "-"
                })
            
            # 单细胞数据
            sc = dist.get('single_cell', {})
            if sc.get('specificity'):
                rna_table_data.append({
                    "样本类型": "🔬 单细胞",
                    "表达特异性": sc['specificity'],
                    "表达量": f"nCPM: {sc['specific_ncpm']}" if sc.get('specific_ncpm') else "-"
                })
            
            # 肿瘤数据
            cancer = dist.get('cancer', {})
            if cancer.get('specificity'):
                rna_table_data.append({
                    "样本类型": "⚕️ 肿瘤",
                    "表达特异性": cancer['specificity'],
                    "表达量": f"pTPM: {cancer['specific_ptpm']}" if cancer.get('specific_ptpm') else "-"
                })
            
            # 血细胞数据
            blood = dist.get('blood', {})
            if blood.get('specificity'):
                rna_table_data.append({
                    "样本类型": "🩸 血细胞",
                    "表达特异性": blood['specificity'],
                    "表达量": f"nTPM: {blood['specific_ntpm']}" if blood.get('specific_ntpm') else "-"
                })
            
            if rna_table_data:
                df_rna = pd.DataFrame(rna_table_data)
                st.table(df_rna)
                
                # 数据解读结论
                st.markdown("**📌 数据解读**")
                interpretations = []
                high_expression_tissues = []
                for row in rna_table_data:
                    spec = row["表达特异性"].lower()
                    if any(x in spec for x in ["high", "highly", "富集", "高"]):
                        high_expression_tissues.append(row["样本类型"])
                
                if high_expression_tissues:
                    interpretations.append(f"该基因在 {', '.join(high_expression_tissues)} 中高表达，提示其在相应组织/细胞类型中可能发挥重要功能。")
                
                # 检查组织特异性程度
                specific_count = sum(1 for row in rna_table_data if any(x in row["表达特异性"].lower() for x in ["specific", "特异性", "富集"]))
                if specific_count == len(rna_table_data):
                    interpretations.append("该基因在所有检测样本类型中均呈现特异性表达模式，属于组织特异性基因。")
                elif specific_count == 0:
                    interpretations.append("该基因表达较为广泛，在各样本类型中均有基础水平表达。")
                else:
                    interpretations.append("该基因呈现部分组织特异性表达，在某些特定组织/细胞类型中表达水平显著更高。")
                
                for interp in interpretations:
                    st.markdown(f"- {interp}")
            else:
                st.info("*暂无RNA表达数据*")
            
            st.divider()
            
            # ===== 第四行：抗体推荐 =====
            st.markdown("#### 🧪 抗体推荐")
            antibody = gene_info.get('antibody', {})
            if antibody.get('name'):
                antibody_name = antibody['name']
                hpa_url = antibody.get('hpa_search_url') or antibody.get('hpa_gene_url', '')
                st.markdown(f"**产品名称:** [{antibody_name}]({hpa_url})")
                st.caption(f"[在HPA数据库中查看抗体详情]({hpa_url})")
            else:
                st.markdown("*暂无抗体推荐*")


# ==================== 报告导出 ====================
class ReportExporter:
    @staticmethod
    def generate_html_report(result: Dict) -> str:
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>慢病毒包装-细胞系评估报告</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                .header {{ background-color: #1f77b4; color: white; padding: 20px; text-align: center; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .pass {{ color: green; }}
                .fail {{ color: red; }}
                .warning {{ color: orange; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>慢病毒包装-细胞系评估报告</h1>
                <p>生成时间: {result.get('timestamp', 'N/A')}</p>
            </div>
            <div class="section">
                <h2>基本信息</h2>
                <p><strong>基因:</strong> {result.get('gene', 'N/A')}</p>
                <p><strong>物种:</strong> {result.get('organism', 'N/A')}</p>
                <p><strong>细胞系:</strong> {result.get('cell_line', 'N/A')}</p>
                <p><strong>实验类型:</strong> {result.get('experiment', 'N/A')}</p>
            </div>
            <div class="section">
                <h2>评估结论</h2>
                <p>{result.get('final_recommendation', 'N/A')}</p>
                <p><strong>依据:</strong> {result.get('primary_basis', 'N/A')}</p>
            </div>
        </body>
        </html>
        """
        return html_content

    @staticmethod
    def generate_csv_report(result: Dict) -> str:
        try:
            df = pd.DataFrame([{
                '基因': result.get('gene', ''),
                '物种': result.get('organism', ''),
                '细胞系': result.get('cell_line', ''),
                '实验类型': result.get('experiment', ''),
                '评估结论': result.get('final_recommendation', ''),
                '评估依据': result.get('primary_basis', ''),
                '生成时间': result.get('timestamp', '')
            }])
            return df.to_csv(index=False)
        except Exception as e:
            return f"Error generating CSV: {str(e)}"

# ==================== 主评估引擎（完整版） ====================
class HybridAssessmentEngine:
    def __init__(self, email: str, ncbi_api_key: Optional[str] = None, ai_api_key: Optional[str] = None):
        self.ncbi = NCBIClient(email, ncbi_api_key)
        self.ai = AIAnalysisClient(ai_api_key) if ai_api_key else None
        self.hard_rules = HybridHardRulesEngine(self.ncbi, self.ai)
        self.hpa = HPADataManager()
        self.hpa_detail_service = HPAGeneDetailService(self.hpa)
        self.email = email

    def assess(self, gene_name: str, organism: str, cell_line: Optional[str],
               experiment_type: str, cell_validation: Optional[Dict] = None) -> Dict:
        """执行完整的混合策略评估（带全局错误处理）"""
        result = {
            'timestamp': datetime.now().isoformat(),
            'gene': gene_name,
            'organism': organism,
            'cell_line': cell_line,
            'experiment': experiment_type,
            'decision_hierarchy': {},
            'final_recommendation': '',
            'primary_basis': '',
            'ai_api_configured': bool(self.ai and self.ai.api_key),
            'errors': [],
            'warnings': [],
            'status': 'running'
        }

        effective_cell_line = None
        if cell_line and str(cell_line).strip():
            effective_cell_line = str(cell_line).strip()
            if cell_validation and cell_validation.get('suggested_standard'):
                effective_cell_line = cell_validation['suggested_standard']
                logger.info(f"使用标准化细胞系名称: {effective_cell_line}")

        if cell_validation:
            result['cell_line_metadata'] = cell_validation

        # 转录本选择
        try:
            with st.spinner("多数据库交叉验证转录本（APPRIS/NCBI/Ensembl，优先NM）..."):
                selector = TranscriptSelector(self.ncbi, self.email)

                gene_info_basic, _ = self.ncbi.fetch_gene_data(gene_name, organism)
                if not gene_info_basic:
                    result['errors'].append(f'无法获取基因 {gene_name} 的信息')
                    result['status'] = 'error'
                    return result

                gene_id = gene_info_basic.get('id')
                if not gene_id:
                    result['errors'].append(f'无法获取基因 {gene_name} 的NCBI Gene ID')
                    result['status'] = 'error'
                    return result
                
                # 更新result中的基因名为官方名称
                official_gene_name = gene_info_basic.get('name', gene_name)
                result['gene'] = official_gene_name

                tx_selection = selector.select_optimal_transcript(
                    gene_name=gene_name,
                    gene_id=gene_id,
                    cell_line=effective_cell_line
                )

                # 处理转录本选择失败的情况
                if not tx_selection:
                    logger.warning("转录本选择返回None，使用空转录本列表继续评估")
                    result['warnings'].append("转录本选择失败，将使用默认设置继续评估")
                    tx_selection = {
                        'selected_transcript': None,
                        'all_transcripts': [],
                        'error': '转录本选择返回None'
                    }

                # 显示详细的诊断信息
                if tx_selection.get('error') or tx_selection.get('diagnosis'):
                    with st.expander("⚠️ 转录本获取失败诊断信息", expanded=True):
                        if tx_selection.get('error'):
                            st.error(f"**错误**: {tx_selection['error']}")
                        
                        if tx_selection.get('diagnosis'):
                            st.write("**各数据库错误详情**:")
                            for db, error in tx_selection['diagnosis'].items():
                                st.error(f"- {db}: {error}")
                        
                        st.markdown("""
                        **可能的原因和解决方案**:
                        1. **网络限制**: Streamlit Cloud 可能无法访问 NCBI/Ensembl API
                        2. **API 配置**: 请在侧边栏配置有效的 NCBI API Key 和邮箱
                        3. **基因ID问题**: 确认基因名称和 Gene ID 正确
                        4. **服务暂时不可用**: 数据库服务可能正在维护
                        
                        **备选方案**:
                        - 检查 NCBI 网站手动获取转录本信息: https://www.ncbi.nlm.nih.gov/gene/
                        - 使用 Ensembl 查看转录本: https://www.ensembl.org/
                        """)
                        
                        # 提供手动输入转录本的选项
                        st.divider()
                        st.write("**手动输入转录本信息（可选）**:")
                        manual_tx_id = st.text_input("转录本 ID (如 NM_001123):", key="manual_tx_id")
                        manual_tx_length = st.number_input("转录本长度 (bp):", min_value=0, max_value=50000, value=0, key="manual_tx_length")
                        
                        if manual_tx_id and manual_tx_length > 0:
                            st.success(f"✓ 将使用手动输入的转录本: {manual_tx_id} ({manual_tx_length}bp)")
                            # 创建手动转录本数据
                            tx_selection['selected_transcript'] = {
                                'id': manual_tx_id,
                                'score': 0.5,
                                'info': {
                                    'source': 'Manual_Input',
                                    'status': 'MANUAL',
                                    'length': manual_tx_length,
                                    'type': 'NM' if manual_tx_id.startswith('NM_') else 'XM',
                                    'title': f'{manual_tx_id} (手动输入)'
                                },
                                'reasons': ['用户手动输入的转录本信息']
                            }
                            tx_selection['all_transcripts'] = [tx_selection['selected_transcript']]
                            tx_selection['error'] = None  # 清除错误状态
                            tx_selection['diagnosis'] = None

                if tx_selection.get('needs_confirmation') and tx_selection.get('conflicts'):
                    st.warning("检测到多个高评分转录本，请选择您需要过表达的特定转录本（已优先选择NM，过滤XM）：")
                    options = []
                    for i, tx in enumerate(tx_selection['conflicts'][:3]):
                        tx_id = tx['id']
                        length = tx['info'].get('length', 0)
                        tx_type = "NM（已验证）" if tx_id.startswith('NM_') else "XM（预测）"
                        reasons = ", ".join(tx['reasons'][:2])
                        options.append(f"{tx_id} ({length}bp) [{tx_type}] - {reasons}")

                    selected = st.radio("选择转录本:", options, key="transcript_select")

                    if selected:
                        selected_tx_id = selected.split()[0]
                        for tx in tx_selection['all_transcripts']:
                            if tx['id'] == selected_tx_id:
                                tx_selection['selected_transcript'] = tx
                                break

                if tx_selection.get('filtered_xm'):
                    st.caption(f"ℹ️ 已自动过滤 {len(tx_selection['filtered_xm'])} 个XM预测转录本")

                selected_tx = tx_selection.get('selected_transcript', {}) or {}

                # 构建转录本列表，处理无转录本数据的情况
                if selected_tx and selected_tx.get('id'):
                    transcripts = [{
                        'id': selected_tx.get('id'),
                        'length': selected_tx.get('info', {}).get('length', 0),
                        'selection_reason': selected_tx.get('reasons', []),
                        'confidence': selected_tx.get('score', 0),
                        'all_candidates': tx_selection.get('all_transcripts', [])
                    }]
                else:
                    transcripts = []
                    logger.warning("无有效转录本数据，使用空列表继续评估")

                gene_info = gene_info_basic

                result['transcript_selection'] = tx_selection
        except Exception as e:
            logger.error(f"转录本选择失败: {e}")
            result['warnings'].append(f"转录本选择失败: {str(e)}，将使用默认设置继续评估")
            # 不返回错误，继续评估
            tx_selection = {
                'selected_transcript': None,
                'all_transcripts': [],
                'error': str(e)
            }
            transcripts = []
            gene_info = gene_info_basic if 'gene_info_basic' in locals() else {}
            result['transcript_selection'] = tx_selection

        # ===== 用户输入转录本号交叉验证 =====
        user_transcript_id = st.session_state.get('user_transcript_id', '').strip()
        if user_transcript_id:
            system_selected = tx_selection.get('selected_transcript', {}) or {}
            system_transcript_id = system_selected.get('id', '') if system_selected else ''
            
            if system_transcript_id and system_transcript_id.upper() == user_transcript_id.upper():
                # 用户输入与系统选择一致
                result['transcript_validation'] = {
                    'status': 'match',
                    'user_input': user_transcript_id,
                    'system_selected': system_transcript_id,
                    'message': f'✓ 用户输入的转录本 {user_transcript_id} 与系统推荐一致'
                }
            elif system_transcript_id:
                # 用户输入与系统选择不一致，提示用户
                result['transcript_validation'] = {
                    'status': 'mismatch',
                    'user_input': user_transcript_id,
                    'system_selected': system_transcript_id,
                    'message': f'⚠️ 用户输入的转录本 {user_transcript_id} 不是最优转录本',
                    'recommendation': f'推荐使用: {system_transcript_id}'
                }
                result['warnings'].append(f"转录本提示：您输入的 {user_transcript_id} 不是最优转录本，推荐使用 {system_transcript_id}")
            else:
                # 系统未返回结果，使用用户输入的转录本
                # 尝试获取用户输入转录本的长度信息
                user_tx_length = None
                try:
                    # 从转录本列表中查找用户输入的转录本
                    for tx in tx_selection.get('all_transcripts', []):
                        if tx.get('id', '').upper() == user_transcript_id.upper():
                            user_tx_length = tx.get('length', 0)
                            break
                    # 如果没找到，尝试从NCBI获取基本信息
                    if not user_tx_length and self.ncbi:
                        # 创建一个基本的转录本信息
                        user_tx_info = {
                            'id': user_transcript_id,
                            'length': 0,  # 未知长度
                            'source': 'user_input'
                        }
                        # 添加到转录本列表
                        if 'all_transcripts' not in tx_selection or not tx_selection['all_transcripts']:
                            tx_selection['all_transcripts'] = []
                        tx_selection['all_transcripts'].append(user_tx_info)
                        # 设置为选中的转录本
                        tx_selection['selected_transcript'] = user_tx_info
                        
                except Exception as e:
                    logger.warning(f"获取用户输入转录本信息失败: {e}")
                
                result['transcript_validation'] = {
                    'status': 'user_fallback',
                    'user_input': user_transcript_id,
                    'system_selected': None,
                    'message': f'使用用户输入的转录本 {user_transcript_id} 进行分析',
                    'note': '系统转录本评估未返回结果，采用用户输入'
                }
                result['warnings'].append(f"使用用户输入的转录本 {user_transcript_id} 进行分析（系统评估未返回结果）")

        result['gene_info'] = {
            'id': gene_info.get('id', ''),
            'name': gene_info.get('name', ''),
            'description': gene_info.get('description', '')[:200]
        }

        # 硬性规则检查
        try:
            with st.spinner("执行混合硬性规则检查（含AI语义分析）..."):
                hard_passed, hard_checks, evidence_summary = self.hard_rules.check_all(
                    gene_name, transcripts, experiment_type
                )
                result['decision_hierarchy']['hard_rules'] = {
                    'passed': hard_passed,
                    'checks': [asdict(c) for c in hard_checks],
                    'evidence_summary': evidence_summary
                }

                blocking = [c for c in hard_checks if not c.passed and not c.overrideable]
                result['blocking_evidence'] = [asdict(c) for c in blocking]

                if blocking:
                    result['is_blocked'] = True
                    result['final_recommendation'] = 'BLOCKED'
                    result['primary_basis'] = '硬性生物学限制'
                    result['warnings'].append('检测到硬性阻断证据，请查看详细报告')
                else:
                    result['is_blocked'] = False
        except Exception as e:
            logger.error(f"硬性规则检查失败: {e}")
            result['warnings'].append(f"硬性规则检查部分失败: {str(e)}")
            hard_checks = []

        # AI基因功能分析
        if self.ai and self.ai.api_key:
            try:
                with st.spinner("AI正在分析基因功能及实验模型数据..."):
                    papers_general = self.ncbi.search_gene_function_literature(gene_name, 'general')
                    papers_oe = self.ncbi.search_gene_function_literature(gene_name, 'overexpression')
                    papers_kd = self.ncbi.search_gene_function_literature(gene_name, 'knockdown')
                    papers_ko = self.ncbi.search_gene_function_literature(gene_name, 'knockout')

                    function_analysis = self.ai.analyze_gene_function_comprehensive(
                        gene_name=gene_name,
                        gene_description=gene_info.get('description', ''),
                        papers_oe=papers_oe,
                        papers_kd=papers_kd,
                        papers_ko=papers_ko,
                        papers_general=papers_general
                    )
                    
                    # 验证 AI 返回的引用是否来自真实检索的文献
                    all_retrieved_pmids = set()
                    pmid_to_paper = {}  # PMID -> paper info mapping
                    for p in papers_general + papers_oe + papers_kd + papers_ko:
                        pmid = p.get('pmid', '')
                        if pmid:
                            all_retrieved_pmids.add(str(pmid))
                            pmid_to_paper[str(pmid)] = p
                    
                    # 过滤 AI 生成的虚假引用，并重新格式化
                    if 'key_references' in function_analysis:
                        valid_refs = []
                        for ref in function_analysis['key_references']:
                            # 尝试提取 PMID
                            import re
                            pmid_match = re.search(r'PMID[:\s]*(\d+)', str(ref))
                            
                            if pmid_match:
                                ref_pmid = pmid_match.group(1)
                                if ref_pmid in all_retrieved_pmids:
                                    # 使用检索到的文献信息重新格式化引用
                                    paper = pmid_to_paper.get(ref_pmid, {})
                                    authors = paper.get('authors', [])
                                    first_author = authors[0].get('name', '').split()[-1] if authors else 'Unknown'
                                    year = paper.get('pubdate', '')[:4] if paper.get('pubdate') else 'N/A'
                                    source = paper.get('source', 'N/A')
                                    title = paper.get('title', '')
                                    formatted_ref = f"{first_author} et al., {year}, {source}, PMID:{ref_pmid}"
                                    if title:
                                        formatted_ref += f" - {title}"
                                    valid_refs.append(formatted_ref)
                                else:
                                    logger.warning(f"AI 生成了未检索到的 PMID: {ref_pmid}")
                                    valid_refs.append(ref + " (数据返回异常，暂时无法提供PMID，需要自行使用标题搜索)")
                            else:
                                # 没有 PMID，尝试模糊匹配标题
                                ref_lower = str(ref).lower()
                                matched = False
                                for pmid, paper in pmid_to_paper.items():
                                    if paper.get('title', '').lower() in ref_lower or ref_lower in paper.get('title', '').lower():
                                        # 匹配成功，使用格式化引用
                                        authors = paper.get('authors', [])
                                        first_author = authors[0].get('name', '').split()[-1] if authors else 'Unknown'
                                        year = paper.get('pubdate', '')[:4] if paper.get('pubdate') else 'N/A'
                                        source = paper.get('source', 'N/A')
                                        title = paper.get('title', '')
                                        formatted_ref = f"{first_author} et al., {year}, {source}, PMID:{pmid}"
                                        if title:
                                            formatted_ref += f" - {title}"
                                        valid_refs.append(formatted_ref)
                                        matched = True
                                        break
                                
                                if not matched:
                                    # 无法匹配，保留原样但标记
                                    valid_refs.append(str(ref)[:100] + "... (数据返回异常，暂时无法提供PMID，需要自行使用标题搜索)")
                        
                        function_analysis['key_references'] = valid_refs
                        function_analysis['references_verified'] = True

                    result['gene_function_analysis'] = {
                        'data': function_analysis,
                        'literature_counts': {
                            'general': len(papers_general),
                            'overexpression': len(papers_oe),
                            'knockdown': len(papers_kd),
                            'knockout': len(papers_ko)
                        },
                        'source': 'AI基于文献综合分析',
                        'status': 'success' if not function_analysis.get('error') else 'error'
                    }
            except Exception as e:
                logger.error(f"基因功能分析失败: {e}")
                result['warnings'].append(f"基因功能分析失败: {str(e)}")
                result['gene_function_analysis'] = {
                    'error': str(e),
                    'status': 'error',
                    'note': 'AI分析过程出错'
                }
        else:
            result['gene_function_analysis'] = {
                'error': '未配置AI API',
                'note': '请在侧边栏配置API Key或在secrets中设置DASHSCOPE_API_KEY',
                'status': 'no_api'
            }

        # HPA基因详细信息查询（仅用于提取基因基本信息，不再查询细胞系特异性表达）
        if organism == 'Homo sapiens':
            try:
                # 检查HPA数据下载状态
                hpa_status = self.hpa.check_and_download()
                
                if hpa_status.get('error'):
                    logger.error(f"【HPA】数据不可用: {hpa_status['error']}")
                    result['hpa_gene_details'] = {
                        'message': f'HPA数据不可用: {hpa_status["error"]}',
                        'note': '请刷新页面重新下载HPA数据',
                        'status': 'download_failed'
                    }
                elif not hpa_status.get('exists'):
                    logger.error("【HPA】数据文件不存在")
                    result['hpa_gene_details'] = {
                        'message': 'HPA数据文件不存在',
                        'note': '请刷新页面下载HPA数据',
                        'status': 'not_found'
                    }
                elif self.hpa_detail_service:
                    with st.spinner("查询HPA基因详细信息..."):
                        # 优先使用用户输入的基因名进行HPA查询（用户输入的通常是基因符号）
                        # 如果失败，再尝试NCBI返回的名称
                        hpa_gene_details = None
                        
                        # 首先尝试用户输入的原始基因名
                        hpa_gene_details = self.hpa_detail_service.get_gene_details(gene_name)
                        
                        # 如果失败且NCBI返回的名称不同，再尝试NCBI名称
                        if (hpa_gene_details and hpa_gene_details.get('error')) or not hpa_gene_details:
                            ncbi_name = gene_info_basic.get('name', '')
                            if ncbi_name and ncbi_name.upper() != gene_name.upper():
                                hpa_gene_details = self.hpa_detail_service.get_gene_details(ncbi_name)
                        
                        if hpa_gene_details and not hpa_gene_details.get('error'):
                            result['hpa_gene_details'] = hpa_gene_details
                        elif hpa_gene_details and hpa_gene_details.get('error'):
                            # 返回了错误信息
                            result['hpa_gene_details'] = {
                                'message': f'HPA查询失败: {hpa_gene_details.get("error")}',
                                'gene_symbol': gene_name,
                                'debug_info': hpa_gene_details,
                                'status': 'query_error'
                            }
                        else:
                            result['hpa_gene_details'] = {
                                'message': f'在HPA数据库中未找到{gene_name}的详细信息',
                                'gene_symbol': gene_name,
                                'status': 'not_found'
                            }
                else:
                    result['hpa_gene_details'] = {
                        'message': 'HPA详细信息服务未初始化',
                        'note': '请检查HPA数据文件是否可用',
                        'status': 'service_not_initialized'
                    }

            except Exception as e:
                logger.error(f"【HPA】基因信息查询失败: {e}")
                import traceback
                logger.error(traceback.format_exc())
                result['warnings'].append(f"HPA基因信息查询失败: {str(e)}")
                result['hpa_analysis_error'] = str(e)
        else:
            result['hpa_gene_details'] = {
                'message': f'HPA仅支持人类基因。当前物种: {organism}',
                'note': 'HPA仅包含人类基因数据'
            }

        # 细胞系评估
        if effective_cell_line:
            try:
                with st.spinner(f"检索 {effective_cell_line} 的相关参数..."):
                    cell_params = self.ncbi.search_cell_lentivirus_params(effective_cell_line)
                    transfection_params = self.ncbi.search_cell_transfection(effective_cell_line)
                    same_cell_studies = self.ncbi.search_same_cell_gene_studies(gene_name, effective_cell_line)
                    cell_culture_papers = self.ncbi.search_cell_culture_literature(effective_cell_line)

                    culture_difficulty = {'error': '未配置AI API或分析失败', 'note': '请配置API'}
                    lv_susceptibility = {'error': '未配置AI API或分析失败', 'note': '请配置API'}

                    if self.ai and self.ai.api_key:
                        try:
                            with st.spinner(f"AI正在分析 {effective_cell_line} 的培养难点..."):
                                culture_difficulty = self.ai.analyze_cell_culture_difficulty(
                                    cell_line=effective_cell_line,
                                    papers=cell_culture_papers if isinstance(cell_culture_papers, list) else []
                                )
                        except Exception as e:
                            logger.error(f"AI培养难点分析失败: {e}")
                            culture_difficulty = {'error': f'AI分析失败: {str(e)}', 'note': 'API调用失败'}

                        try:
                            lv_susceptibility = self.ai.analyze_lentivirus_susceptibility(
                                cell_line=effective_cell_line,
                                papers=cell_params if isinstance(cell_params, list) else []
                            )
                        except Exception as e:
                            logger.error(f"AI易感性分析失败: {e}")
                            lv_susceptibility = {'error': f'AI分析失败: {str(e)}', 'note': 'API调用失败'}

                    result['cell_assessment'] = {
                        'lentivirus_params': cell_params if cell_params else [],
                        'transfection_params': transfection_params if transfection_params else [],
                        'same_cell_gene_studies': same_cell_studies if same_cell_studies else [],
                        'culture_difficulty': culture_difficulty,
                        'lentivirus_susceptibility': lv_susceptibility,
                        'cell_line_searched': effective_cell_line,
                        'status': 'success' if not culture_difficulty.get('error') else 'partial'
                    }
            except Exception as e:
                logger.error(f"细胞系评估失败: {e}")
                result['warnings'].append(f"细胞系评估失败: {str(e)}")
                result['cell_assessment'] = {
                    'error': str(e),
                    'cell_line_searched': effective_cell_line,
                    'status': 'error'
                }
        else:
            result['cell_assessment'] = {
                'skipped': True,
                'reason': '未输入有效细胞系名称',
                'status': 'skipped'
            }

        # 序列设计
        if experiment_type.lower() in ['knockdown', 'knockout']:
            if experiment_type.lower() == 'knockdown':
                # siRNA/shRNA从文献检索（仅使用Materials and Methods中报道的序列，不使用AI设计）
                try:
                    with st.spinner("正在从PubMed文献Materials and Methods中检索已报道的siRNA/shRNA序列..."):
                        design_data = self.ai.design_rnai_sequences(
                            gene_name=gene_name,
                            gene_id=gene_info.get('id', ''),
                            gene_description=gene_info.get('description', '')
                        )
                        result['sequence_designs'] = {
                            'type': 'siRNA/shRNA (文献Methods检索)',
                            'designs': design_data,
                            'source': 'PubMed文献Materials and Methods检索（仅使用文献报道序列，无AI推测）',
                            'status': 'success' if not design_data.get('error') and design_data.get('sequences') else 'no_data'
                        }
                except Exception as e:
                    logger.error(f"siRNA文献检索失败: {e}")
                    result['sequence_designs'] = {
                        'type': 'siRNA/shRNA检索',
                        'designs': {'error': str(e)},
                        'source': '检索失败',
                        'status': 'error'
                    }
            else:
                # CRISPR sgRNA从文献检索（仅使用Materials and Methods中报道的序列，不使用AI设计）
                try:
                    with st.spinner("正在从PubMed文献Materials and Methods中检索已报道的sgRNA序列..."):
                        design_data = self.ai.design_crispr_sequences(
                            gene_name=gene_name,
                            gene_id=gene_info.get('id', ''),
                            gene_description=gene_info.get('description', '')
                        )
                        result['sequence_designs'] = {
                            'type': 'sgRNA (文献Methods检索)',
                            'designs': design_data,
                            'source': 'PubMed文献Materials and Methods检索（仅使用文献报道序列，无AI推测）',
                            'status': 'success' if not design_data.get('error') and design_data.get('sgrnas') else 'no_data'
                        }
                except Exception as e:
                    logger.error(f"sgRNA文献检索失败: {e}")
                    result['sequence_designs'] = {
                        'type': 'sgRNA检索',
                        'designs': {'error': str(e)},
                        'source': '检索失败',
                        'status': 'error'
                    }

        # 最终推荐
        try:
            if not result.get('final_recommendation'):
                warning_checks = [c for c in hard_checks if not c.passed and c.overrideable]
                if warning_checks:
                    result['final_recommendation'] = "警告：检测到潜在风险，建议谨慎操作"
                    result['primary_basis'] = f"基于{len(warning_checks)}项警告（可人工覆盖）"
                else:
                    result['final_recommendation'] = "未检测到明确风险，可进行标准流程"
                    result['primary_basis'] = "基于核心数据库筛查和文献检索"
        except Exception as e:
            logger.error(f"生成最终推荐失败: {e}")
            result['final_recommendation'] = "评估完成，但生成推荐时出错"
            result['primary_basis'] = "部分评估数据可用"

        result['status'] = 'success' if not result['errors'] else 'partial'
        return result


# ==================== UI渲染（完整版） ====================
def render_sidebar():
    with st.sidebar:
        st.header("API配置")

        st.subheader("NCBI配置")
        ncbi_email = st.text_input("NCBI邮箱", value="", key="ncbi_email_input",
                                  help="优先使用此处输入，留空使用Secrets")
        ncbi_key = st.text_input("NCBI API Key", type="password", key="ncbi_key_input",
                                help="可选，用于提高访问频率限制")

        email, key, error = APIConfig.get_ncbi_credentials()
        if error:
            st.error(error)
        else:
            st.success("NCBI API有效")

        st.divider()

        st.subheader("AI配置（可选，但强烈建议配置）")
        qwen_key = st.text_input("通义千问API Key", type="password", key="qwen_key_input",
                                help="可选，用于AI文献语义分析、细胞培养难点分析（序列设计仅使用文献Methods数据，无需AI）")

        final_qwen = APIConfig.get_qwen_api_key()
        if final_qwen:
            st.success("✓ AI API已配置 - 将启用智能细胞系评估和基因功能分析")
        else:
            st.warning("✗ 未配置AI API（请在上方输入或在Secrets中设置DASHSCOPE_API_KEY）")
            st.caption("注：序列设计从文献Methods检索，不依赖AI")

        st.divider()
        st.caption("核心列表+文献补充+AI语义分析混合策略")
        
        # 添加系统诊断按钮
        st.divider()
        st.subheader("🔧 系统诊断")
        if st.button("测试API连接", key="test_api_btn"):
            with st.spinner("正在测试API连接..."):
                # 测试NCBI连接
                try:
                    ncbi_test = NCBIClient(email=email, api_key=key)
                    # 尝试搜索一个简单的查询
                    test_result = ncbi_test._make_request('esearch.fcgi', {
                        'db': 'pubmed',
                        'term': 'test',
                        'retmode': 'json',
                        'retmax': 1
                    })
                    if test_result:
                        st.success("✓ NCBI API 连接正常")
                    else:
                        st.error("✗ NCBI API 返回空结果")
                except Exception as e:
                    st.error(f"✗ NCBI API 连接失败: {str(e)[:100]}")
                
                # 测试AI连接
                if final_qwen:
                    try:
                        ai_test = QwenAnalyzer(api_key=final_qwen)
                        # 尝试一个简单的API调用
                        test_response = ai_test._make_request({
                            'model': 'qwen-turbo',
                            'input': {
                                'messages': [
                                    {'role': 'user', 'content': 'Hello'}
                                ]
                            }
                        })
                        if test_response and test_response.get('output'):
                            st.success("✓ AI API 连接正常")
                        else:
                            st.error("✗ AI API 返回异常")
                    except Exception as e:
                        st.error(f"✗ AI API 连接失败: {str(e)[:100]}")
                else:
                    st.warning("⚠️ 未配置AI API，跳过测试")

def render_main_panel():
    st.markdown("""
        <h1 style='text-align: center; color: #1f77b4; margin-bottom: 30px;'>
        慢病毒包装-细胞系评估系统
        </h1>
    """, unsafe_allow_html=True)

    st.markdown("### 实验参数输入")

    if 'cell_line_validation' not in st.session_state:
        st.session_state.cell_line_validation = None
    if 'final_cell_line' not in st.session_state:
        st.session_state.final_cell_line = None
    if 'assessment_locked' not in st.session_state:
        st.session_state.assessment_locked = False
    if 'cell_line_selected' not in st.session_state:
        st.session_state.cell_line_selected = None
    if 'cell_line_input' not in st.session_state:
        st.session_state.cell_line_input = ''

    col1, col2 = st.columns(2)

    # 检查是否处于锁定状态
    is_locked = st.session_state.get('assessment_locked', False)
    
    if is_locked:
        st.warning("🔒 当前评估已锁定。如需进行新评估，请点击下方的「开始新评估」按钮。")

    with col1:
        organism = st.selectbox(
            "物种",
            ["human", "mouse", "rat", "cho", "pig", "monkey"],
            format_func=lambda x: {
                "human": "人类 (Homo sapiens)",
                "mouse": "小鼠 (Mus musculus)",
                "rat": "大鼠 (Rattus norvegicus)",
                "cho": "CHO (Cricetulus griseus)",
                "pig": "家猪 (Sus scrofa)",
                "monkey": "猴子 (Macaca mulatta)"
            }.get(x, x),
            disabled=is_locked
        )

        # ===== HPA基因自动补全服务初始化 =====
        if 'hpa_gene_service' not in st.session_state:
            # 尝试从现有的HPA管理器获取数据文件路径
            hpa_manager = st.session_state.get('hpa_manager')
            if hpa_manager:
                # 检查HPA数据是否可用
                hpa_status = hpa_manager.check_and_download()
                if hpa_status.get('error'):
                    st.warning(f"⚠️ HPA数据下载失败: {hpa_status['error']}")
                    st.info("📌 请刷新页面重新尝试下载")
                elif not hpa_status.get('exists') and not hpa_status.get('downloaded'):
                    st.warning("⚠️ HPA数据尚未下载")
                    st.info("📌 首次使用需要下载约200MB数据，请刷新页面")
            
            st.session_state['hpa_gene_service'] = HPAGeneAutocompleteService(hpa_manager)
            st.session_state['hpa_gene_detail_service'] = HPAGeneDetailService(hpa_manager)
        
        hpa_gene_service = st.session_state.get('hpa_gene_service')
        hpa_detail_service = st.session_state.get('hpa_gene_detail_service')
        
        gene_service = GeneAutocompleteService()  # 保留用于其他可能的用途
        # 基因输入完全基于HPA数据包（抛弃NCBI自动补全）
        gene_component = GeneInputComponent(hpa_gene_service, hpa_detail_service)
        gene = gene_component.render(organism, key_prefix="main_gene", disabled=is_locked)
        
        # ===== 转录本号输入（可选，用于交叉验证）=====
        user_transcript_id = st.text_input(
            "转录本号（可选，用于交叉验证）",
            value=st.session_state.get('user_transcript_id', ''),
            placeholder="例如：NM_001101.3, XM_017012345.1...",
            help="输入RefSeq转录本号，系统将与自动评估的最优转录本进行交叉验证",
            key="user_transcript_input",
            disabled=is_locked
        )
        st.session_state['user_transcript_id'] = user_transcript_id
        if user_transcript_id:
            st.caption(f"✓ 已输入转录本号: {user_transcript_id}")

    with col2:
        # ===== HPA细胞系自动补全输入（新增功能）=====
        if 'cell_line_component' not in st.session_state:
            st.session_state.cell_line_component = HPACellLineAutocompleteService()

        cell_service = st.session_state.cell_line_component

        # 输入框 - 根据锁定状态禁用
        cell_input = st.text_input(
            "细胞系（HPA数据库自动补全，输入1字符即显示建议）",
            value=st.session_state.get('cell_line_input', ''),
            placeholder="例如：A549, HeLa, MCF7, NCI-H226, HEK293, SH-SY5Y...",
            help="输入细胞系名称，系统从HPA数据库1206个细胞系中匹配。支持模糊匹配、大小写不敏感、分隔符容错。",
            key="cell_line_widget",
            disabled=is_locked
        )

        # 保存输入并触发建议更新
        if cell_input != st.session_state.get('cell_line_input', ''):
            st.session_state['cell_line_input'] = cell_input
            st.session_state['cell_line_selected'] = None
            if len(cell_input) >= 1:
                safe_rerun()

        cell_line_value = None
        cell_metadata = None

        # 显示自动补全建议（调试功能已移除）
        if cell_input and len(cell_input) >= 1 and not st.session_state.get('cell_line_selected'):
            suggestions = cell_service.get_suggestions(cell_input, limit=8)

            if suggestions:
                st.caption(f"💡 HPA数据库匹配 ({len(suggestions)}个建议)：")
                cols = st.columns(min(len(suggestions), 4))

                for i, sug in enumerate(suggestions):
                    with cols[i % 4]:
                        # 精确匹配显示特殊标记
                        is_exact = sug['match_type'] == 'exact'
                        label = f"✓ {sug['display_name']}" if is_exact else sug['display_name']
                        btn_type = "primary" if is_exact else "secondary"
                        help_text = f"匹配类型: {sug['match_type']}, 分数: {sug['score']}"

                        if st.button(label, key=f"cell_sug_{i}", use_container_width=True,
                                   type=btn_type, help=help_text):
                            st.session_state['cell_line_selected'] = sug['hpa_name']
                            st.session_state['cell_line_input'] = sug['hpa_name']
                            st.session_state['cell_line_validation'] = {
                                'input': cell_input,
                                'normalized': cell_service._normalize(sug['hpa_name']),
                                'is_valid': True,
                                'hpa_standard_name': sug['hpa_name'],
                                'suggested_standard': sug['hpa_name'],
                                'confidence': 1.0,
                                'match_type': sug['match_type'],
                                'needs_confirmation': False,
                                'warning': None
                            }
                            safe_rerun()

            elif not suggestions and cell_input:
                st.caption(f"💡 未找到匹配的HPA细胞系 (输入: '{cell_input}')")
                st.info("提示：可继续输入或点击「开始评估」使用自定义名称")

            # 检查是否有精确匹配但名称不同
            exact_match = cell_service.get_exact_match(cell_input)
            if exact_match and exact_match.upper() != cell_input.upper():
                st.info(f"💡 检测到HPA标准名称: **{exact_match}** (与您输入的 '{cell_input}' 略有不同)")
                col_yes, col_no = st.columns([1.5, 2])
                with col_yes:
                    if st.button(f"✓ 使用 {exact_match}", key="use_hpa_exact", type="primary"):
                        st.session_state['cell_line_selected'] = exact_match
                        st.session_state['cell_line_input'] = exact_match
                        st.session_state['cell_line_validation'] = {
                            'input': cell_input,
                            'normalized': cell_service._normalize(exact_match),
                            'is_valid': True,
                            'hpa_standard_name': exact_match,
                            'suggested_standard': exact_match,
                            'confidence': 0.95,
                            'match_type': 'exact',
                            'needs_confirmation': False
                        }
                        safe_rerun()
                with col_no:
                    if st.button("保持原输入", key="keep_original"):
                        st.session_state['cell_line_selected'] = cell_input
                        st.session_state['cell_line_validation'] = {
                            'input': cell_input,
                            'normalized': cell_service._normalize(cell_input),
                            'is_valid': False,
                            'hpa_standard_name': None,
                            'suggested_standard': cell_input,
                            'confidence': 0.5,
                            'match_type': 'none',
                            'needs_confirmation': True,
                            'warning': f"'{cell_input}' 不是标准HPA细胞系名称"
                        }
                        safe_rerun()

            cell_line_value = cell_input
            cell_metadata = st.session_state.get('cell_line_validation', {})

        # 显示已选择状态
        if st.session_state.get('cell_line_selected'):
            cell_line_value = st.session_state['cell_line_selected']
            st.success(f"✓ 已选择HPA标准细胞系: **{cell_line_value}**")

            # 显示细胞类型提示
            cell_type = CellLineNormalizer.get_cell_type_hint(cell_line_value)
            if cell_type:
                st.caption(f"细胞类型: {cell_type}")

        # 保存到session state供后续使用
        if cell_line_value:
            st.session_state.final_cell_line = cell_line_value
            st.session_state.cell_line_validation = cell_metadata if cell_metadata else {
                'input': cell_input,
                'is_valid': cell_service.is_valid_cell_line(cell_line_value) if cell_line_value else False,
                'hpa_standard_name': cell_line_value if cell_service.is_valid_cell_line(cell_line_value) else None
            }

    exp_type = st.selectbox(
        "评估选项",
        ["overexpression", "knockdown", "knockout"],
        format_func=lambda x: {
            "overexpression": "过表达 (OE)",
            "knockdown": "敲低 (RNAi)",
            "knockout": "敲除 (CRISPR)"
        }.get(x, x),
        disabled=is_locked
    )

    # 按钮区域
    col_btn1, col_btn2 = st.columns([3, 1])
    
    with col_btn1:
        if is_locked:
            analyze = st.button("🔒 评估已锁定", type="primary", use_container_width=True, disabled=True)
        else:
            analyze = st.button("开始AI智能评估", type="primary", use_container_width=True)
    
    with col_btn2:
        if is_locked:
            if st.button("🔄 开始新评估", type="secondary", use_container_width=True):
                # 重置锁定状态和所有相关session state
                st.session_state.assessment_locked = False
                st.session_state.cell_line_input = ''
                st.session_state.cell_line_selected = None
                st.session_state.cell_line_validation = None
                st.session_state.final_cell_line = None
                st.session_state.last_assessment_result = None  # 清除保存的评估结果
                # 清除之前的评估结果
                for key in list(st.session_state.keys()):
                    if key.startswith('gene_input_') or key.startswith('hpa_info_'):
                        del st.session_state[key]
                safe_rerun()
        else:
            st.button("🔄 开始新评估", type="secondary", use_container_width=True, disabled=True, help="请先完成当前评估")

    final_cell_line = st.session_state.get('final_cell_line')
    return organism, gene, final_cell_line, exp_type, analyze

def render_results(result: Dict):
    """渲染评估结果（带完整错误处理）"""
    # 首先显示错误和警告
    if result.get('errors'):
        st.error("### ❌ 评估过程中发生错误")
        for error in result['errors']:
            st.error(f"- {error}")

    if result.get('warnings'):
        st.warning("### ⚠️ 警告")
        for warning in result['warnings']:
            st.warning(f"- {warning}")

    # 如果状态是错误，提前返回
    if result.get('status') == 'error':
        st.info("💡 建议：请检查输入参数是否正确，或稍后重试。如问题持续，请联系管理员。")
        return

    if 'error' in result:
        st.error(result['error'])
        return

    st.divider()
    st.markdown(f"## 评估报告 - {html.escape(result['gene'])}")

    if result.get('ai_api_configured'):
        st.success("✓ AI API已激活 - 所有功能可用")
    else:
        st.warning("✗ AI API未配置 - 部分功能受限（基因功能分析、序列设计、细胞培养难点分析需要API）")

    cell_meta = result.get('cell_line_metadata')
    if cell_meta and cell_meta.get('is_valid'):
        normalized = cell_meta.get('normalized', '')
        input_val = cell_meta.get('input', '')
        if normalized != input_val:
            with st.expander("细胞系名称标准化信息", expanded=False):
                st.write(f"**原始输入**: {input_val}")
                st.write(f"**标准化名称**: {cell_meta.get('suggested_standard', normalized)}")
                st.write(f"**置信度**: {cell_meta.get('confidence', 0):.0%}")
                if cell_meta.get('cell_type'):
                    st.write(f"**细胞类型**: {cell_meta['cell_type']}")

    col_exp1, col_exp2, col_exp3 = st.columns(3)
    with col_exp1:
        exporter = ReportExporter()
        html_report = exporter.generate_html_report(result)
        st.download_button(
            "导出HTML报告",
            html_report,
            file_name=f"assessment_{result['gene']}_{datetime.now().strftime('%Y%m%d')}.html",
            mime="text/html"
        )
    with col_exp2:
        csv_report = exporter.generate_csv_report(result)
        st.download_button(
            "导出CSV报告",
            csv_report,
            file_name=f"assessment_{result['gene']}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    with col_exp3:
        # PDF导出 - 在新标签页打开
        pdf_html = exporter.generate_html_report(result)  # 先用HTML作为基础
        # 使用JavaScript在新标签页打开
        b64_pdf = base64.b64encode(pdf_html.encode()).decode()
        pdf_link = f'<a href="data:text/html;base64,{b64_pdf}" target="_blank" style="text-decoration:none;"><button style="width:100%;padding:8px 16px;background-color:#f0f2f6;border:1px solid #ddd;border-radius:4px;cursor:pointer;">在新标签页查看报告</button></a>'
        st.markdown(pdf_link, unsafe_allow_html=True)

    rec = result['final_recommendation']
    rec_color = {"BLOCKED": "#ffebee", "警告": "#fff3e0", "未检测": "#fff8e1", "未": "#e8f5e9"}.get(rec[:2], "#f5f5f5")

    st.markdown(f"""
        <div style='padding: 20px; background-color: {rec_color};
        border-radius: 10px; text-align: center; margin: 20px 0;
        border: 2px solid #ddd;'>
            <h3>{html.escape(rec)}</h3>
            <small>{html.escape(result.get('primary_basis', ''))}</small>
            {"<br><small style='color:red;'>检测到硬性阻断证据，但已完成其他分析</small>" if result.get('is_blocked') else ""}
        </div>
    """, unsafe_allow_html=True)

    # 修改后的标签页列表
    tabs = st.tabs([
        "慢病毒包装可行性评估",
        "基因功能分析",
        "HPA基因与蛋白信息",     # 整合蛋白基础功能的HPA信息
        "细胞系评估",
        "序列设计",
        "转录本选择"
    ])

    with tabs[0]:
        st.markdown("### 混合硬性规则检查")
        
        # ===== CDS长度信息 =====
        tx_data = result.get('transcript_selection', {})
        tx_validation = result.get('transcript_validation', {})
        if tx_data and isinstance(tx_data, dict):
            selected = tx_data.get('selected_transcript', {})
            
            with st.container():
                st.subheader("📏 转录本/CDS信息")
                
                # 显示用户输入转录本验证信息
                if tx_validation:
                    val_status = tx_validation.get('status', '')
                    user_tx = tx_validation.get('user_input', '')
                    system_tx = tx_validation.get('system_selected', '')
                    
                    if val_status == 'match':
                        st.success(f"✓ 用户输入转录本 {user_tx} 与系统推荐一致")
                    elif val_status == 'mismatch':
                        st.warning(f"⚠️ 用户输入: {user_tx}")
                        st.info(f"💡 系统推荐: {system_tx}")
                    elif val_status == 'user_fallback':
                        st.info(f"ℹ️ 使用用户输入转录本: {user_tx}")
                        st.caption("系统评估未返回结果，采用用户输入")
                
                # 检查是否有错误
                if tx_data.get('error'):
                    st.error(f"**转录本ID**: 异常，无返回数据")
                    st.markdown(f"**序列长度**: 无返回数据")
                    st.caption(f"原因: {tx_data.get('error')}")
                elif selected:
                    tx_id = selected.get('id', '')
                    tx_info = selected.get('info', {})
                    tx_length = tx_info.get('length', 0)
                    
                    if tx_id and tx_length and tx_id != f"{result.get('gene', '')}_UNKNOWN":
                        ncbi_url = f"https://www.ncbi.nlm.nih.gov/nuccore/{tx_id}"
                        st.markdown(f"**转录本ID**: [{tx_id}]({ncbi_url})")
                        st.markdown(f"**序列长度**: {tx_length} bp")
                        st.caption(f"[→ NCBI查看详情]({ncbi_url})")
                    else:
                        st.error(f"**转录本ID**: 异常，无返回数据")
                        st.markdown(f"**序列长度**: 无返回数据")
                else:
                    st.error(f"**转录本ID**: 异常，无返回数据")
                    st.markdown(f"**序列长度**: 无返回数据")
            st.divider()
        
        hierarchy = result.get('decision_hierarchy', {})
        hard_rules = hierarchy.get('hard_rules', {})
        evidence_summary = hard_rules.get('evidence_summary', {})

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("核心数据库命中", len(evidence_summary.get('core_hits', [])))
        with col2:
            st.metric("文献补充发现", len(evidence_summary.get('literature_hits', [])))
        with col3:
            st.metric("总检查项", len(hard_rules.get('checks', [])))
        with col4:
            ai_status = "是" if evidence_summary.get('ai_analyzed') else "否"
            st.metric("AI语义分析", ai_status)

        st.divider()

        for check in hard_rules.get('checks', []):
            icon = "✅" if check['passed'] else "⛔" if not check['overrideable'] else "⚠️"
            color = "green" if check['passed'] else "red" if not check['overrideable'] else "orange"
            level_badge = "核心" if check.get('check_level') == "core" else "文献"

            evidence_md = ""
            if check.get('evidence_papers'):
                evidence_md = "<br/><small>证据文献：</small><br/>"
                for paper in check['evidence_papers'][:2]:
                    pmid = paper.get('pmid', '')
                    title = paper.get('title', '')[:80]
                    ai_info = ""
                    if 'ai_confidence' in paper:
                        ai_info = f"<br/><small>AI置信度: {paper['ai_confidence']:.0%}</small>"
                    match_info = ""
                    if 'match_score' in paper:
                        match_info = f" [匹配:{paper['match_score']}]"
                    evidence_md += f'<small>• <a href="https://pubmed.ncbi.nlm.nih.gov/{pmid}/" target="_blank">PMID:{pmid}</a> {html.escape(title)}...{match_info}</small>{ai_info}<br/>'

            pmid_list = check.get('pmid_list', [])
            pmid_badge = ""
            if pmid_list:
                pmid_badge = f'<br/><small>PMID: {", ".join([f"<a href=\'https://pubmed.ncbi.nlm.nih.gov/{p}/\' target=\'_blank\'>{p}</a>" for p in pmid_list[:3]])}</small>'

            st.markdown(f"""
                <div style='padding: 15px; border-left: 4px solid {color};
                background-color: #f8f9fa; margin: 10px 0; border-radius: 5px;'>
                    <h4>{icon} {html.escape(check['rule_name'])} {level_badge}</h4>
                    <p>{html.escape(check['reason'])}</p>
                    <small>来源: {html.escape(check['source'])}</small>
                    {pmid_badge}
                    {evidence_md}
                </div>
            """, unsafe_allow_html=True)

    with tabs[1]:
        st.markdown("### 基因功能全面分析")
        func_analysis = result.get('gene_function_analysis')

        if func_analysis and isinstance(func_analysis, dict):
            status = func_analysis.get('status', 'unknown')

            if status == 'no_api':
                st.warning("⚠️ 未配置AI API，无法提供基因功能分析")
                st.info("请在侧边栏输入通义千问API Key，或在Secrets中设置DASHSCOPE_API_KEY")
            elif status == 'error':
                st.error(f"❌ 基因功能分析失败: {func_analysis.get('error', '未知错误')}")
                if func_analysis.get('note'):
                    st.info(func_analysis['note'])
            elif status == 'success':
                data = func_analysis.get('data', {})
                lit_counts = func_analysis.get('literature_counts', {})

                st.markdown("#### 文献覆盖统计")
                cols = st.columns(4)
                cols[0].metric("基因功能文献", lit_counts.get('general', 0))
                cols[1].metric("过表达研究", lit_counts.get('overexpression', 0))
                cols[2].metric("敲低研究", lit_counts.get('knockdown', 0))
                cols[3].metric("敲除研究", lit_counts.get('knockout', 0))
                st.caption(f"来源: {func_analysis.get('source', 'AI分析')}")
                
                st.divider()
                
                # 检查是否有任何功能数据（不包括蛋白基础功能，已整合到HPA部分）
                has_any_data = any([
                    'overexpression' in data and data['overexpression'],
                    'knockdown' in data and data['knockdown'],
                    'knockout' in data and data['knockout'],
                    'disease_relevance' in data and data['disease_relevance']
                ])
                
                if not has_any_data:
                    st.warning("⚠️ AI分析未返回具体功能数据")
                    st.info("可能原因：1) 检索到的文献数量不足 2) AI返回格式异常 3) 该基因研究较少")
                
                st.info("ℹ️ **蛋白基础功能**已整合至「HPA基因与蛋白信息」标签页，以HPA数据为优先展示")

                if 'overexpression' in data and data['overexpression']:
                    with st.expander("📈 过表达效应"):
                        oe = data['overexpression']
                        if 'cell_models' in oe and oe['cell_models']:
                            st.markdown("**细胞模型:**")
                            for model in oe['cell_models']:
                                st.markdown(f"**• {model.get('cell_line', 'N/A')}**")
                                st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;表型: {model.get('phenotype', 'N/A')}")
                                st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;机制: {model.get('mechanism', 'N/A')}")
                                # 改进文献展示
                                ref = model.get('reference', 'N/A')
                                if ref and ref != 'N/A':
                                    # 尝试提取PMID
                                    import re
                                    pmid_match = re.search(r'PMID[:\s]*(\d+)', str(ref))
                                    if pmid_match:
                                        pmid = pmid_match.group(1)
                                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;文献: [{ref[:80]}...](https://pubmed.ncbi.nlm.nih.gov/{pmid}/) (PMID:{pmid})")
                                    else:
                                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;文献: {ref[:100]}{'...' if len(ref) > 100 else ''}")
                                st.markdown("---")
                        if 'summary' in oe and oe['summary']:
                            st.markdown(f"> **总结**: {oe['summary']}")

                if 'knockout' in data and data['knockout']:
                    with st.expander("🧪 敲除效应"):
                        ko = data['knockout']
                        if 'cell_models' in ko and ko['cell_models']:
                            st.markdown("**细胞模型:**")
                            for model in ko['cell_models']:
                                st.markdown(f"**• {model.get('cell_line', 'N/A')}** ({model.get('method', '')})")
                                st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;表型: {model.get('phenotype', 'N/A')}")
                                st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;细胞活力: {model.get('viability', 'N/A')}")
                                # 改进文献展示
                                ref = model.get('reference', 'N/A')
                                if ref and ref != 'N/A':
                                    import re
                                    pmid_match = re.search(r'PMID[:\s]*(\d+)', str(ref))
                                    if pmid_match:
                                        pmid = pmid_match.group(1)
                                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;文献: [{ref[:80]}...](https://pubmed.ncbi.nlm.nih.gov/{pmid}/) (PMID:{pmid})")
                                    else:
                                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;文献: {ref[:100]}{'...' if len(ref) > 100 else ''}")
                                st.markdown("---")
                        if 'summary' in ko and ko['summary']:
                            st.markdown(f"> **总结**: {ko['summary']}")

                if 'knockdown' in data and data['knockdown']:
                    with st.expander("🔻 敲低效应"):
                        kd = data['knockdown']
                        if 'cell_models' in kd and kd['cell_models']:
                            st.markdown("**细胞模型:**")
                            for model in kd['cell_models']:
                                st.markdown(f"**• {model.get('cell_line', 'N/A')}** ({model.get('method', '')})")
                                st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;表型: {model.get('phenotype', 'N/A')}")
                                # 改进文献展示
                                ref = model.get('reference', 'N/A')
                                if ref and ref != 'N/A':
                                    import re
                                    pmid_match = re.search(r'PMID[:\s]*(\d+)', str(ref))
                                    if pmid_match:
                                        pmid = pmid_match.group(1)
                                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;文献: [{ref[:80]}...](https://pubmed.ncbi.nlm.nih.gov/{pmid}/) (PMID:{pmid})")
                                    else:
                                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;文献: {ref[:100]}{'...' if len(ref) > 100 else ''}")
                                st.markdown("---")
                        if 'summary' in kd and kd['summary']:
                            st.markdown(f"> **总结**: {kd['summary']}")

                if 'disease_relevance' in data and data['disease_relevance']:
                    with st.expander("🏥 疾病相关性"):
                        dr = data['disease_relevance']
                        st.markdown(f"**肿瘤作用**: {dr.get('cancer', 'N/A')}")
                        st.markdown(f"**其他疾病**: {dr.get('other_diseases', 'N/A')}")
                        st.markdown(f"**治疗潜力**: {dr.get('therapeutic_potential', 'N/A')}")

                if 'key_references' in data and data['key_references']:
                    with st.expander("📚 关键参考文献"):
                        st.caption("⚠️ **注意**：以下引用基于AI分析的文献生成")
                        if data.get('references_verified'):
                            st.success("✓ 引用已与检索到的文献匹配")
                        
                        for ref in data['key_references']:
                            # 尝试提取PMID并添加链接
                            import re
                            pmid_match = re.search(r'PMID[:\s]*(\d+)', str(ref))
                            if pmid_match:
                                pmid = pmid_match.group(1)
                                st.markdown(f"- [{ref}](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)")
                            else:
                                st.markdown(f"- {ref}")

                if 'experimental_notes' in data and data['experimental_notes']:
                    st.markdown(f"> **实验设计建议**: {data['experimental_notes']}")
            else:
                st.info("正在加载基因功能分析数据...")
        else:
            st.info("未获取到基因功能分析数据")

    # ==================== HPA基因与蛋白信息（整合蛋白基础功能）====================
    with tabs[2]:
        st.markdown("### 🧬 HPA基因与蛋白信息")
        st.caption("数据来源: Human Protein Atlas (HPA) + 文献蛋白功能数据（HPA数据优先）")
        
        # 优先使用评估结果中已保存的HPA数据
        gene_details = result.get('hpa_gene_details')
        gene_name = result.get('gene') or result.get('gene_name')
        
        # 如果没有保存的HPA数据，尝试实时查询（向后兼容）
        if not gene_details and gene_name:
            try:
                hpa_detail_service = st.session_state.get('hpa_gene_detail_service')
                if hpa_detail_service:
                    gene_details = hpa_detail_service.get_gene_details(gene_name)
            except Exception as e:
                logger.warning(f"实时获取HPA基因详情失败: {e}")
        
        if gene_details and isinstance(gene_details, dict):
            # 检查是否是错误状态
            if gene_details.get('error') or gene_details.get('status') in ['download_failed', 'not_found', 'query_error']:
                st.warning(f"HPA数据不可用: {gene_details.get('message', gene_details.get('error', '未知错误'))}")
            else:
                data = gene_details
                
                # ========== 基础信息表格（Ensembl ID、Uniprot ID、基因组位置、蛋白定位与功能、抗体推荐）==========
                st.subheader("📋 基因基础信息")
                
                # 准备表格数据
                table_data = []
                
                # Ensembl ID
                ensembl_id = data.get('ensembl_id', 'N/A')
                ensembl_url = data.get('ensembl_url', '')
                ensembl_display = f"[{ensembl_id}]({ensembl_url})" if ensembl_id != 'N/A' and ensembl_url else ensembl_id
                table_data.append({"信息项": "Ensembl ID", "内容": ensembl_display})
                
                # Uniprot ID
                uniprot_id = data.get('uniprot_id', 'N/A')
                uniprot_url = data.get('uniprot_url', '')
                uniprot_display = f"[{uniprot_id}]({uniprot_url})" if uniprot_id != 'N/A' and uniprot_url else uniprot_id
                table_data.append({"信息项": "Uniprot ID", "内容": uniprot_display})
                
                # 基因组位置
                genome_loc = data.get('genome_location', 'N/A')
                chromosome = data.get('chromosome', '')
                position = data.get('position', '')
                ensembl_id = data.get('ensembl_id', '')
                if genome_loc != 'N/A':
                    if chromosome and position:
                        pos_parts = position.replace(',', '').split('-')
                        start_pos = pos_parts[0] if pos_parts else position.replace(',', '')
                        ucsc_url = f"https://genome.ucsc.edu/cgi-bin/hgTracks?db=hg38&position=chr{chromosome}:{start_pos}"
                    else:
                        ucsc_url = f"https://genome.ucsc.edu/cgi-bin/hgGene?hgg_gene={ensembl_id}"
                    genome_display = f"`{genome_loc}` [→ UCSC]({ucsc_url})"
                else:
                    genome_display = "N/A"
                table_data.append({"信息项": "基因组位置", "内容": genome_display})
                
                # 蛋白定位与功能
                loc = data.get('protein_localization', {})
                func = data.get('protein_function', {})
                protein_info_parts = []
                if loc.get('subcellular_main'):
                    protein_info_parts.append(f"主要定位: {loc['subcellular_main']}")
                if loc.get('subcellular_additional'):
                    protein_info_parts.append(f"附加定位: {loc['subcellular_additional']}")
                if loc.get('secretome_location'):
                    protein_info_parts.append(f"分泌位置: {loc['secretome_location']}")
                if func.get('biological_process'):
                    protein_info_parts.append(f"生物过程: {func['biological_process']}")
                if func.get('molecular_function'):
                    protein_info_parts.append(f"分子功能: {func['molecular_function']}")
                protein_display = "; ".join(protein_info_parts) if protein_info_parts else "N/A"
                table_data.append({"信息项": "蛋白定位与功能", "内容": protein_display})
                
                # 抗体推荐
                antibody = data.get('antibody', {})
                if antibody.get('name'):
                    antibody_name = antibody['name']
                    hpa_url = antibody.get('hpa_search_url') or antibody.get('hpa_gene_url', '')
                    antibody_display = f"[{antibody_name}]({hpa_url})" if hpa_url else antibody_name
                else:
                    antibody_display = "N/A"
                table_data.append({"信息项": "抗体推荐", "内容": antibody_display})
                
                # 显示表格 - 使用 st.table 避免依赖 tabulate
                df_basic = pd.DataFrame(table_data)
                st.table(df_basic)
                
                st.divider()
                
                # ========== 蛋白基础功能（整合AI分析与HPA数据，HPA优先）==========
                st.subheader("🧬 蛋白基础功能")
                
                # 获取AI分析的蛋白基础功能数据
                func_analysis = result.get('gene_function_analysis', {})
                ai_protein_func = {}
                if func_analysis and func_analysis.get('status') == 'success':
                    ai_data = func_analysis.get('data', {})
                    ai_protein_func = ai_data.get('protein_function', {})
                
                # HPA蛋白功能数据
                hpa_protein_func = data.get('protein_function', {})
                hpa_protein_loc = data.get('protein_localization', {})
                
                # 整合展示（HPA数据优先）
                protein_func_data = []
                
                # 蛋白类别 - HPA没有此字段，使用AI数据
                category = ai_protein_func.get('category', 'N/A')
                if category and category != 'N/A':
                    protein_func_data.append({"功能项": "蛋白类别", "内容": category, "来源": "文献分析"})
                
                # 生物过程 - HPA优先
                biological_process = hpa_protein_func.get('biological_process', '')
                if biological_process:
                    protein_func_data.append({"功能项": "生物过程", "内容": biological_process, "来源": "HPA"})
                elif ai_protein_func.get('pathways'):
                    protein_func_data.append({"功能项": "生物过程/通路", "内容": ai_protein_func['pathways'], "来源": "文献分析"})
                
                # 分子功能 - HPA优先
                molecular_function = hpa_protein_func.get('molecular_function', '')
                if molecular_function:
                    protein_func_data.append({"功能项": "分子功能", "内容": molecular_function, "来源": "HPA"})
                
                # 亚细胞定位 - HPA优先
                subcellular_main = hpa_protein_loc.get('subcellular_main', '')
                subcellular_add = hpa_protein_loc.get('subcellular_additional', '')
                loc_parts = []
                if subcellular_main:
                    loc_parts.append(f"主要: {subcellular_main}")
                if subcellular_add:
                    loc_parts.append(f"附加: {subcellular_add}")
                if loc_parts:
                    protein_func_data.append({"功能项": "亚细胞定位", "内容": "; ".join(loc_parts), "来源": "HPA"})
                elif ai_protein_func.get('cellular_location'):
                    protein_func_data.append({"功能项": "亚细胞定位", "内容": ai_protein_func['cellular_location'], "来源": "文献分析"})
                
                # 结构域 - AI数据（HPA无此字段）
                domains = ai_protein_func.get('domains', '')
                if domains and domains != 'N/A':
                    protein_func_data.append({"功能项": "结构域", "内容": domains, "来源": "文献分析"})
                
                # 疾病相关性 - HPA优先
                disease_involvement = hpa_protein_func.get('disease_involvement', '')
                if disease_involvement:
                    protein_func_data.append({"功能项": "疾病相关性", "内容": disease_involvement, "来源": "HPA"})
                elif ai_protein_func.get('disease_relevance'):
                    protein_func_data.append({"功能项": "疾病相关性", "内容": ai_protein_func['disease_relevance'], "来源": "文献分析"})
                
                # 组织表达 - HPA优先
                tissue_expression = data.get('rna_expression', {}).get('tissue_specificity', '')
                if tissue_expression:
                    protein_func_data.append({"功能项": "组织表达特异性", "内容": tissue_expression, "来源": "HPA"})
                elif ai_protein_func.get('tissue_expression'):
                    protein_func_data.append({"功能项": "组织表达", "内容": ai_protein_func['tissue_expression'], "来源": "文献分析"})
                
                if protein_func_data:
                    df_protein = pd.DataFrame(protein_func_data)
                    st.table(df_protein)
                    st.caption("📌 数据整合原则：HPA数据优先，缺失项补充文献分析数据")
                else:
                    st.info("*暂无蛋白基础功能数据*")
                
                st.divider()
                
                # ========== RNA表达数据（整合后的单一表格）==========
                st.subheader("📊 RNA表达数据")
                
                # 收集所有RNA表达数据
                rna = data.get('rna_expression', {})
                dist = data.get('rna_distribution', {})
                
                # 构建RNA表达表格数据
                rna_table_data = []
                
                # 组织数据（合并 tissue 和 rna_expression 中的组织特异性）
                tissue = dist.get('tissue', {})
                tissue_specificity = tissue.get('specificity') or rna.get('tissue_specificity', '')
                tissue_ntpm = tissue.get('specific_ntpm') or rna.get('tissue_specific_ntpm', '')
                if tissue_specificity:
                    rna_table_data.append({
                        "样本类型": "🧬 组织",
                        "表达特异性": tissue_specificity,
                        "表达量": f"nTPM: {tissue_ntpm}" if tissue_ntpm else "-"
                    })
                
                # 单细胞数据
                sc = dist.get('single_cell', {})
                if sc.get('specificity'):
                    rna_table_data.append({
                        "样本类型": "🔬 单细胞",
                        "表达特异性": sc['specificity'],
                        "表达量": f"nCPM: {sc['specific_ncpm']}" if sc.get('specific_ncpm') else "-"
                    })
                
                # 肿瘤数据
                cancer = dist.get('cancer', {})
                if cancer.get('specificity'):
                    rna_table_data.append({
                        "样本类型": "⚕️ 肿瘤",
                        "表达特异性": cancer['specificity'],
                        "表达量": f"pTPM: {cancer['specific_ptpm']}" if cancer.get('specific_ptpm') else "-"
                    })
                
                # 血细胞数据
                blood = dist.get('blood', {})
                if blood.get('specificity'):
                    rna_table_data.append({
                        "样本类型": "🩸 血细胞",
                        "表达特异性": blood['specificity'],
                        "表达量": f"nTPM: {blood['specific_ntpm']}" if blood.get('specific_ntpm') else "-"
                    })
                
                if rna_table_data:
                    df_rna = pd.DataFrame(rna_table_data)
                    st.table(df_rna)
                    
                    # 数据解读结论
                    st.markdown("**📌 数据解读**")
                    interpretations = []
                    high_expression_tissues = []
                    for row in rna_table_data:
                        spec = row["表达特异性"].lower()
                        if any(x in spec for x in ["high", "highly", "富集", "高"]):
                            high_expression_tissues.append(row["样本类型"])
                    
                    if high_expression_tissues:
                        interpretations.append(f"该基因在 {', '.join(high_expression_tissues)} 中高表达，提示其在相应组织/细胞类型中可能发挥重要功能。")
                    
                    # 检查组织特异性程度
                    specific_count = sum(1 for row in rna_table_data if any(x in row["表达特异性"].lower() for x in ["specific", "特异性", "富集"]))
                    if specific_count == len(rna_table_data):
                        interpretations.append("该基因在所有检测样本类型中均呈现特异性表达模式，属于组织特异性基因。")
                    elif specific_count == 0:
                        interpretations.append("该基因表达较为广泛，在各样本类型中均有基础水平表达。")
                    else:
                        interpretations.append("该基因呈现部分组织特异性表达，在某些特定组织/细胞类型中表达水平显著更高。")
                    
                    for interp in interpretations:
                        st.markdown(f"- {interp}")
                else:
                    st.info("*暂无RNA表达数据*")
        else:
            st.info("未获取到HPA基因信息（请确保输入有效基因名称）")

    # ==================== 细胞系评估 ====================

    with tabs[3]:
        st.markdown("### 细胞系评估数据")
        cell_data = result.get('cell_assessment')
        if cell_data and isinstance(cell_data, dict):
            status = cell_data.get('status', 'unknown')

            if status == 'skipped':
                st.warning(f"未进行细胞系评估：{cell_data.get('reason', '未输入细胞系名称')}")
            elif status == 'error':
                st.error(f"细胞系评估出错：{cell_data.get('error')}")
            else:
                searched_cell = cell_data.get('cell_line_searched', 'Unknown')
                st.success(f"当前分析细胞系：**{searched_cell}**")

                if result.get('ai_api_configured'):
                    st.success("✓ AI分析已执行")
                else:
                    st.warning("⚠️ 未配置AI API，培养难点和易感性分析受限")

                studies = cell_data.get('same_cell_gene_studies', [])
                if studies:
                    st.subheader("同细胞系同基因文献")
                    for study in studies[:5]:
                        st.markdown(f"""
                            - **{html.escape(study.get('title', ''))}**
                            *{html.escape(study.get('journal', ''))}* ({study.get('year', '')})
                            [PMID: {study.get('pmid', '')}]({study.get('url', '')})
                        """)
                else:
                    st.info("未检索到该细胞系中当前基因的共同研究文献")

                st.divider()

                st.subheader("细胞培养难点清单")
                culture_diff = cell_data.get('culture_difficulty', {})
                if culture_diff and isinstance(culture_diff, dict):
                    if culture_diff.get('error'):
                        st.error(f"❌ 获取培养难点分析失败：{culture_diff.get('error')}")
                    else:
                        if 'data_source_note' in culture_diff:
                            note = culture_diff['data_source_note']
                            if '文献' in note:
                                st.success(f"📚 {note}")
                            else:
                                st.info(f"🤖 {note}")

                        # 精简后的展示逻辑
                        has_any = False
                        
                        # 1. 气体/环境敏感 - 只显示非5% CO2、非37°C时提示实际的培养条件，不然显示"37°C，5% CO2"
                        env_items = culture_diff.get('environment', [])
                        env_display = []
                        # 检查是否有特殊环境要求
                        has_special_env = False
                        for item in env_items:
                            # 排除常规条件
                            if any(x in item for x in ['5% CO₂', '37℃', '37 °C', '37°C']):
                                continue
                            if '低氧' in item or '非5%' in item or '非37' in item or '震荡' in item or 'pH' in item:
                                env_display.append(item)
                                has_special_env = True
                        
                        with st.expander("🌡️ 气体/环境敏感", expanded=False):
                            if has_special_env:
                                for item in env_display:
                                    st.markdown(f"<div style='margin: 5px 0; padding: 8px; background-color: #e3f2fd; border-radius: 5px; border-left: 3px solid #2196F3;'>{html.escape(str(item))}</div>", unsafe_allow_html=True)
                                has_any = True
                            else:
                                st.markdown("<div style='margin: 5px 0; padding: 8px; background-color: #f5f5f5; border-radius: 5px;'>37°C，5% CO₂</div>", unsafe_allow_html=True)
                        
                        # 2. 操作复杂度 - 只有非0.25%胰酶消化时提示
                        op_items = culture_diff.get('operation', [])
                        op_display = []
                        for item in op_items:
                            # 排除0.25%胰酶（常规条件）
                            if '0.25%胰酶' in item or '0.25% 胰酶' in item:
                                continue
                            op_display.append(item)
                        
                        with st.expander("🔧 操作复杂度", expanded=False):
                            if op_display:
                                for item in op_display:
                                    st.markdown(f"<div style='margin: 5px 0; padding: 8px; background-color: #fff3e0; border-radius: 5px; border-left: 3px solid #FF9800;'>{html.escape(str(item))}</div>", unsafe_allow_html=True)
                                has_any = True
                            else:
                                st.markdown("<div style='margin: 5px 0; padding: 8px; background-color: #f5f5f5; border-radius: 5px;'>常规操作，无特殊要求</div>", unsafe_allow_html=True)
                        
                        # 3. 时间/成本 - 只有细胞增殖速度低（文献中特别提到增殖慢、倍增时间超过48小时）时提示
                        tc_items = culture_diff.get('time_cost', [])
                        tc_display = []
                        for item in tc_items:
                            # 只保留明确提到增殖慢、倍增时间>48小时的条目
                            if any(x in item for x in ['>48小时', '超过48小时', '倍增时间', '生长极其缓慢', '增殖慢', '生长缓慢']):
                                tc_display.append(item)
                        
                        with st.expander("⏱️ 时间/成本", expanded=False):
                            if tc_display:
                                for item in tc_display:
                                    st.markdown(f"<div style='margin: 5px 0; padding: 8px; background-color: #e8f5e9; border-radius: 5px; border-left: 3px solid #4CAF50;'>{html.escape(str(item))}</div>", unsafe_allow_html=True)
                                has_any = True
                            else:
                                st.markdown("<div style='margin: 5px 0; padding: 8px; background-color: #f5f5f5; border-radius: 5px;'>常规生长速度，无特殊要求</div>", unsafe_allow_html=True)
                        
                        # 4. 关键警告 - 支原体属于常规管控项，不做提示。否则反馈"无特殊要求"
                        warn_items = culture_diff.get('special_warnings', [])
                        warn_display = []
                        for item in warn_items:
                            # 排除支原体相关（常规管控项）
                            if '支原体' in item:
                                continue
                            warn_display.append(item)
                        
                        with st.expander("⚠️ 关键警告", expanded=True):  # 关键警告默认展开
                            if warn_display:
                                for item in warn_display:
                                    st.markdown(f"<div style='margin: 5px 0; padding: 8px; background-color: #ffcdd2; border-radius: 5px; border-left: 3px solid #f44336;'><strong>{html.escape(str(item))}</strong></div>", unsafe_allow_html=True)
                                has_any = True
                            else:
                                st.markdown("<div style='margin: 5px 0; padding: 8px; background-color: #f5f5f5; border-radius: 5px;'>无特殊要求</div>", unsafe_allow_html=True)
                        
                        # 5. Protocol建议 - 只有提示必须每天换液、每天传代时提示。否则反馈"无特殊要求"
                        proto_items = culture_diff.get('protocol_tips', [])
                        proto_display = []
                        for item in proto_items:
                            # 只保留必须每天换液/传代的提示
                            if any(x in item for x in ['每天换液', '每天传代', '每日换液', '每日传代', '必须每天']):
                                proto_display.append(item)
                        
                        with st.expander("💡 Protocol建议", expanded=False):
                            if proto_display:
                                for item in proto_display:
                                    st.markdown(f"<div style='margin: 5px 0; padding: 8px; background-color: #e0f7fa; border-radius: 5px; border-left: 3px solid #00BCD4;'>{html.escape(str(item))}</div>", unsafe_allow_html=True)
                                has_any = True
                            else:
                                st.markdown("<div style='margin: 5px 0; padding: 8px; background-color: #f5f5f5; border-radius: 5px;'>无特殊要求</div>", unsafe_allow_html=True)
                        
                        # 培养基和基质要求保持不变（始终显示，因为这些是重要的信息）
                        cm_items = culture_diff.get('culture_medium', [])
                        if cm_items:
                            has_any = True
                            with st.expander("🧫 培养基特殊要求", expanded=False):
                                for item in cm_items:
                                    st.markdown(f"<div style='margin: 5px 0; padding: 8px; background-color: #ffebee; border-radius: 5px; border-left: 3px solid #E91E63;'>{html.escape(str(item))}</div>", unsafe_allow_html=True)
                        
                        coat_items = culture_diff.get('coating_matrix', [])
                        if coat_items:
                            has_any = True
                            with st.expander("🔲 基质/包被要求", expanded=False):
                                for item in coat_items:
                                    st.markdown(f"<div style='margin: 5px 0; padding: 8px; background-color: #f3e5f5; border-radius: 5px; border-left: 3px solid #9C27B0;'>{html.escape(str(item))}</div>", unsafe_allow_html=True)

                        if not has_any:
                            st.success("✓ 未检索到该细胞系的特殊培养难点，可能为常规培养细胞")

                        if culture_diff.get('verified_by'):
                            with st.expander("支持文献/依据"):
                                for ref in culture_diff['verified_by']:
                                    st.markdown(f"- {ref}")
                else:
                    st.warning("未获取到培养难点数据格式异常")

                st.divider()

                st.subheader("慢病毒易感性分析")
                lv_susc = cell_data.get('lentivirus_susceptibility', {})
                if lv_susc and isinstance(lv_susc, dict):
                    if lv_susc.get('error'):
                        st.error(f"❌ 易感性分析失败：{lv_susc.get('error')}")
                    else:
                        if 'cell_line_info' in lv_susc:
                            st.info(f"**细胞系信息**: {lv_susc['cell_line_info']}")

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            level = lv_susc.get('susceptibility_level', 'Unknown')
                            st.metric("易感性等级", level)
                        with col2:
                            st.metric("推荐MOI", lv_susc.get('recommended_moi', 'N/A'))
                        with col3:
                            st.metric("典型感染效率", lv_susc.get('infection_efficiency', 'N/A'))

                        if lv_susc.get('challenges'):
                            with st.expander("感染挑战"):
                                for challenge in lv_susc['challenges']:
                                    st.markdown(f"- {challenge}")

                        if lv_susc.get('optimization_tips'):
                            with st.expander("优化建议"):
                                for tip in lv_susc['optimization_tips']:
                                    st.markdown(f"- {tip}")
                else:
                    st.warning("未获取到易感性数据")
        else:
            st.info("未获取到细胞系评估数据")

    # ==================== 序列设计 ====================
    with tabs[4]:
        st.markdown("### 序列设计参考")
        seq_data = result.get('sequence_designs')
        if seq_data and isinstance(seq_data, dict):
            status = seq_data.get('status', 'unknown')

            if status == 'no_api':
                st.warning("⚠️ NCBI API未配置，无法检索文献序列")
                st.info("请在侧边栏输入NCBI邮箱和API Key")
            elif status == 'error':
                st.error(f"❌ 序列检索失败: {seq_data.get('designs', {}).get('error', '未知错误')}")
            elif status == 'no_data':
                st.warning("⚠️ 未在文献Materials and Methods中找到序列")
                st.info("建议：1) 查阅文献补充材料；2) 使用在线工具设计")
            else:
                st.subheader(f"{seq_data.get('type', '')}")
                st.caption(f"来源: {seq_data.get('source', '')}")

                designs = seq_data.get('designs', {})
                if isinstance(designs, dict):
                    if designs.get('error'):
                        st.error(f"设计错误: {designs.get('error')}")
                    else:
                        if 'sequences' in designs:
                            for i, seq in enumerate(designs.get('sequences', [])):
                                with st.expander(f"候选序列 #{i+1}: {seq.get('target_seq', 'N/A')[:20]}..."):
                                    st.write(f"**靶序列**: `{seq.get('target_seq', 'N/A')}`")
                                    st.write(f"**靶标区域**: {seq.get('target_region', 'N/A')}")
                                    st.write(f"**设计原理**: {seq.get('design_rationale', 'N/A')}")
                                    st.write(f"**预期效率**: {seq.get('efficiency_score', 'N/A')}")
                                    refs = seq.get('references', [])
                                    if refs:
                                        st.write("**参考文献**:")
                                        st.caption("⚠️ 以下引用由AI生成，请通过PubMed核实其真实性")
                                        for ref in refs:
                                            st.markdown(f"- *{ref.get('title', '')}* ({ref.get('year', '')}) [{ref.get('pmid_or_patent', '')}]({ref.get('url', '')})")

                            if 'shrna_vector_design' in designs:
                                with st.expander("shRNA载体设计建议"):
                                    vd = designs['shrna_vector_design']
                                    st.write(f"**Loop序列**: {vd.get('loop_sequence', 'N/A')}")
                                    st.write(f"**启动子**: {vd.get('promoter', 'N/A')}")
                                    st.write(f"**克隆位点**: {vd.get('cloning_sites', 'N/A')}")
                                    st.write(f"**备注**: {vd.get('notes', 'N/A')}")

                            if 'notes' in designs:
                                st.info(f"**注意事项**: {designs['notes']}")

                        elif 'sgrnas' in designs:
                            # 显示数据来源
                            source = designs.get('source', 'unknown')
                            search_loc = designs.get('search_location', '')
                            if source == 'literature_and_patents':
                                st.info(f"📚 sgRNA序列来源: {search_loc}")
                            
                            sgrnas = designs.get('sgrnas', [])
                            supplementary_only = designs.get('supplementary_only', [])
                            patent_metadata = designs.get('patent_metadata', [])
                            
                            # 分离文献和专利来源的序列
                            literature_sgrnas = [s for s in sgrnas if s.get('source') == 'literature']
                            patent_sgrnas = [s for s in sgrnas if s.get('source') == 'patent']
                            
                            if not sgrnas and not supplementary_only and not patent_metadata:
                                st.warning("⚠️ 未在文献Methods或专利全文中找到sgRNA序列")
                                if 'alternative_tools' in designs:
                                    st.subheader("建议使用以下工具设计：")
                                    for tool in designs.get('alternative_tools', []):
                                        st.markdown(f"- [{tool.get('name', 'Tool')}]({tool.get('url', '#')})")
                            else:
                                # 显示Methods中找到的序列（文献）
                                if literature_sgrnas:
                                    st.success(f"✅ 在{len(literature_sgrnas)}篇文献的Methods中找到sgRNA序列")
                                    for i, sgrna in enumerate(literature_sgrnas):
                                        with st.expander(f"sgRNA #{i+1}: {sgrna.get('sequence', 'N/A')}"):
                                            st.write(f"**靶序列**: `{sgrna.get('sequence', 'N/A')}`")
                                            st.write(f"**PAM**: {sgrna.get('pam', 'NGG')}")
                                            st.write(f"**说明**: {sgrna.get('target_exon', 'N/A')}")
                                            
                                            # 显示文献来源
                                            ref = sgrna.get('reference', {})
                                            if ref:
                                                st.write("**来源**: ")
                                                st.markdown(f"- *{ref.get('title', '')}* ({ref.get('year', '')}) [{ref.get('pmid_or_patent', '')}]({ref.get('url', '')})")
                                                if ref.get('pmc_url'):
                                                    st.markdown(f"- [PMC全文]({ref.get('pmc_url')})")
                                
                                # 显示专利中找到的序列
                                if patent_sgrnas:
                                    st.success(f"📜 在{len(patent_sgrnas)}件专利的权利要求书/详细说明中找到sgRNA序列")
                                    for i, sgrna in enumerate(patent_sgrnas):
                                        with st.expander(f"专利sgRNA #{i+1}: {sgrna.get('sequence', 'N/A')}"):
                                            st.write(f"**靶序列**: `{sgrna.get('sequence', 'N/A')}`")
                                            st.write(f"**PAM**: {sgrna.get('pam', 'NGG')}")
                                            st.write(f"**说明**: {sgrna.get('target_exon', 'N/A')}")
                                            
                                            # 显示专利来源
                                            ref = sgrna.get('reference', {})
                                            if ref:
                                                st.write("**来源**: ")
                                                st.markdown(f"- *{ref.get('title', '')}* ({ref.get('year', '')})")
                                                st.markdown(f"- 专利号: {ref.get('pmid_or_patent', 'N/A')}")
                                                if ref.get('url'):
                                                    st.markdown(f"- [Google Patents]({ref.get('url')})")
                                
                                # 显示相关专利（需查阅全文）
                                if patent_metadata:
                                    st.info(f"📜 发现{len(patent_metadata)}件相关专利（可能包含sgRNA序列）")
                                    with st.expander("查看专利列表（需查阅全文确认）"):
                                        for i, pat in enumerate(patent_metadata):
                                            st.markdown(f"**{i+1}. {pat.get('title', '')}** ({pat.get('year', '')})")
                                            st.markdown(f"   - 专利号: {pat.get('patent_id', 'N/A')}")
                                            st.caption(f"   💡 {pat.get('note', '')}")
                                            if pat.get('url'):
                                                st.markdown(f"   - [Google Patents]({pat.get('url')})")
                                
                                # 显示只在补充材料中的文献
                                if supplementary_only:
                                    st.info(f"📎 另外发现{len(supplementary_only)}篇文献的sgRNA序列在补充材料中")
                                    with st.expander("查看这些文献（需手动查阅补充材料）"):
                                        for i, sup in enumerate(supplementary_only):
                                            st.markdown(f"**{i+1}. {sup.get('title', '')}** ({sup.get('year', '')})")
                                            st.markdown(f"   - [PubMed]({sup.get('url', '')}) | [PMC全文]({sup.get('pmc_url', '')})")
                                            st.caption(f"   💡 {sup.get('note', '')}")
                            
                            # 显示注意事项
                            if 'notes' in designs:
                                st.info(f"💡 **注意事项**: {designs['notes']}")

                            if 'lentivirus_vector' in designs:
                                with st.expander("慢病毒载体建议"):
                                    lv = designs['lentivirus_vector']
                                    st.write(f"**骨架**: {lv.get('backbone', 'N/A')}")
                                    st.write(f"**启动子**: {lv.get('promoter', 'N/A')}")
                                    st.write(f"**筛选标记**: {lv.get('selection', 'N/A')}")
                                    st.write(f"**克隆策略**: {lv.get('cloning_strategy', 'N/A')}")

                            if 'validation_method' in designs:
                                st.success(f"**验证方法**: {designs['validation_method']}")
                else:
                    st.error("序列设计数据格式异常")
        else:
            st.info("敲低和敲除实验可查看序列设计建议（从文献检索已报道的sgRNA）")

    # ==================== 转录本选择 ====================
    with tabs[5]:
        st.markdown("### 转录本选择详情（多数据库交叉验证）")
        tx_data = result.get('transcript_selection', {})
        
        # ===== 显示用户输入转录本验证结果 =====
        tx_validation = result.get('transcript_validation', {})
        if tx_validation:
            val_status = tx_validation.get('status', '')
            if val_status == 'match':
                st.success(f"✓ {tx_validation.get('message', '')}")
            elif val_status == 'mismatch':
                st.warning(f"⚠️ {tx_validation.get('message', '')}")
                st.info(f"💡 {tx_validation.get('recommendation', '')}")
            elif val_status == 'user_fallback':
                st.info(f"ℹ️ {tx_validation.get('message', '')}")
                if tx_validation.get('note'):
                    st.caption(tx_validation['note'])
        
        if tx_data and isinstance(tx_data, dict):
            if not tx_data.get('error'):
                try:
                    cols = st.columns(len(tx_data.get('database_coverage', {})))
                    for i, (db, count) in enumerate(tx_data.get('database_coverage', {}).items()):
                        cols[i].metric(f"{db}数据", f"{count}个转录本")
                except:
                    pass

                if tx_data.get('filtered_xm'):
                    st.info(f"ℹ️ 已自动过滤 {len(tx_data['filtered_xm'])} 个XM预测转录本，优先展示NM已验证转录本")
                    with st.expander("查看被过滤的XM转录本"):
                        st.write(tx_data['filtered_xm'])

                st.divider()

                selected = tx_data.get('selected_transcript', {})
                if selected:
                    st.write("**选中转录本评分依据：**")
                    for reason in selected.get('reasons', []):
                        st.markdown(f"- {reason}")

                    st.write(f"**综合评分**: {selected.get('score', 0):.2f}")
                    st.write(f"**转录本ID**: {selected.get('id', 'N/A')}")
                    tx_info = selected.get('info', {})
                    if tx_info.get('length'):
                        st.write(f"**序列长度**: {tx_info['length']}bp")

                    sources = tx_info.get('sources', [])
                    if sources:
                        st.caption(f"数据来源: {', '.join(sources)}")
                    if tx_info.get('ncbi_status'):
                        st.caption(f"NCBI状态: {tx_info['ncbi_status']}")
                    if tx_info.get('appris_label'):
                        st.caption(f"APPRIS标签: {tx_info['appris_label']}")
                else:
                    st.error("**转录本ID**: 异常，无返回数据")
                    st.markdown("**序列长度**: 无返回数据")

                if tx_data.get('conflicts'):
                    st.warning("⚠️ 存在其他高分候选，建议确认：")
                    for conflict in tx_data['conflicts'][1:3]:
                        tx_type = "NM（已验证）" if conflict['id'].startswith('NM_') else "XM（预测）" if conflict['id'].startswith('XM_') else "其他"
                        st.write(f"- {conflict['id']} [{tx_type}] (评分: {conflict['score']:.2f})")
            else:
                st.error(f"转录本选择出错: {tx_data.get('error')}")
        else:
            st.info("未获取到转录本选择数据")

def main():
    # ===== 日志收集器（用于主界面显示） =====
    if 'app_logs' not in st.session_state:
        st.session_state['app_logs'] = []
    
    # 自定义日志处理器，将日志显示在主界面
    class StreamlitLogHandler(logging.Handler):
        def emit(self, record):
            try:
                msg = self.format(record)
                if len(st.session_state['app_logs']) > 100:
                    st.session_state['app_logs'].pop(0)
                st.session_state['app_logs'].append({
                    'time': datetime.now().strftime('%H:%M:%S'),
                    'level': record.levelname,
                    'message': msg
                })
            except:
                pass
    
    # 添加 Streamlit 日志处理器
    if not any(isinstance(h, StreamlitLogHandler) for h in logger.handlers):
        st_handler = StreamlitLogHandler()
        st_handler.setLevel(logging.INFO)
        st_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(st_handler)
    
    try:
        if not AuthManager.check_password():
            st.stop()
    except Exception as e:
        st.error(f"密码验证模块错误: {e}")
        st.stop()

    # ===== HPA 数据下载（带状态检测） =====
    hpa_status = {'exists': False, 'downloaded': False, 'error': None}
    try:
        hpa_manager = HPADataManager()
        hpa_status = hpa_manager.check_and_download()
        st.session_state['hpa_manager'] = hpa_manager
        
        # 显示下载状态
        if hpa_status.get('error'):
            st.error(f"⚠️ HPA数据下载失败: {hpa_status['error']}")
            st.warning("📌 请刷新页面重新尝试下载")
        elif hpa_status.get('downloaded'):
            st.success("✓ HPA数据下载完成")
        elif hpa_status.get('exists'):
            logger.info("【HPA】数据文件已存在，跳过下载")
            
    except Exception as e:
        st.warning(f"HPA数据管理器初始化警告: {e}")
        hpa_status['error'] = str(e)

    try:
        render_sidebar()
    except Exception as e:
        st.error(f"侧边栏渲染错误: {e}")

    try:
        organism, gene, cell_line, exp_type, analyze = render_main_panel()
    except Exception as e:
        st.error(f"主面板渲染错误: {e}")
        return

    # 检查是否有已保存的评估结果（锁定状态下恢复显示）
    if st.session_state.get('assessment_locked') and st.session_state.get('last_assessment_result'):
        st.success("✓ 显示上次评估结果（页面已锁定）")
        render_results(st.session_state['last_assessment_result'])

    if analyze:
        if not gene:
            st.error("请输入或选择一个基因")
            return

        is_valid, error_msg = SecurityConfig.validate_gene_name(gene)
        if not is_valid:
            st.error(f"输入验证失败: {error_msg}")
            return

        gene_clean = SecurityConfig.sanitize_input(gene, 50)

        organism_map = {
            "human": "Homo sapiens",
            "mouse": "Mus musculus",
            "rat": "Rattus norvegicus",
            "cho": "Cricetulus griseus",
            "pig": "Sus scrofa",
            "monkey": "Macaca mulatta"
        }
        organism_clean = organism_map.get(organism, organism)

        try:
            email, ncbi_key, error = APIConfig.get_ncbi_credentials()
            if error:
                st.error(error)
                return
        except Exception as e:
            st.error(f"API配置错误: {e}")
            return

        try:
            ai_key = APIConfig.get_qwen_api_key()
        except Exception as e:
            st.warning(f"AI API配置警告: {e}")
            ai_key = None

        cell_validation = st.session_state.get('cell_line_validation')

        try:
            engine = HybridAssessmentEngine(
                email=email,
                ncbi_api_key=ncbi_key,
                ai_api_key=ai_key
            )

            with st.spinner("正在进行混合策略评估..."):
                result = engine.assess(
                    gene_clean,
                    organism_clean,
                    cell_line,
                    exp_type,
                    cell_validation=cell_validation
                )
                # 保存评估结果到 session_state
                st.session_state['last_assessment_result'] = result
                render_results(result)
                
                # 评估完成后锁定页面
                st.session_state.assessment_locked = True
                st.success("✓ 评估完成！页面已锁定。如需进行新评估，请点击「开始新评估」按钮。")
                safe_rerun()

        except Exception as e:
            logger.exception(f"Unhandled error: {e}")
            st.error(f"系统错误: {str(e)}")
            st.exception(e)
    
    # ===== 底部：日志显示区域 =====
    with st.expander("📋 应用日志（点击展开）", expanded=False):
        if st.session_state['app_logs']:
            # 复制按钮
            log_text = "\n".join([f"[{log['time']}] {log['level']}: {log['message']}" 
                                  for log in st.session_state['app_logs']])
            st.text_area("日志内容（可复制）", log_text, height=200, key="log_display")
            
            col1, col2 = st.columns([1, 5])
            with col1:
                if st.button("🗑️ 清空日志"):
                    st.session_state['app_logs'] = []
                    st.rerun()
        else:
            st.caption("暂无日志")

if __name__ == "__main__":
    main()
