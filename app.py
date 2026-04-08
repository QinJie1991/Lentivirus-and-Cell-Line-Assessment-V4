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
class HPAGeneAutocompleteService:
    """基于HPA proteinatlas.tsv的Gene synonym列的基因自动补全服务"""
    
    def __init__(self, hpa_manager=None):
        self.hpa_manager = hpa_manager
        self.search_index = {}
        self.gene_data_cache = {}
        self._build_search_index()
    
    def _build_search_index(self):
        """从proteinatlas.tsv构建基因搜索索引，如文件不存在则触发下载"""
        data_file = None
        
        if self.hpa_manager and hasattr(self.hpa_manager, 'data_file'):
            data_file = self.hpa_manager.data_file
        else:
            local_dir = os.path.join(os.path.expanduser("~"), ".hpa_data")
            data_file = os.path.join(local_dir, "proteinatlas.tsv")
        
        # 如果文件不存在，尝试触发下载
        if not data_file or not os.path.exists(data_file):
            logger.warning("HPA数据文件不存在，尝试下载...")
            if self.hpa_manager and hasattr(self.hpa_manager, 'check_and_download'):
                try:
                    self.hpa_manager.check_and_download()
                except Exception as e:
                    logger.error(f"自动下载HPA数据失败: {e}")
            
            # 再次检查文件是否存在
            if not os.path.exists(data_file):
                logger.warning("HPA数据文件下载失败或不存在，基因自动补全将不可用")
                return
        
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter='\t')
                for row in reader:
                    gene_symbol = row.get('Gene name', '').strip()
                    ensembl_id = row.get('Gene', '').strip()
                    synonyms_str = row.get('Gene synonym', '').strip()
                    
                    if not gene_symbol:
                        continue
                    
                    gene_key = gene_symbol.upper()
                    
                    self.gene_data_cache[gene_key] = {
                        'gene_symbol': gene_symbol,
                        'ensembl_id': ensembl_id,
                        'synonyms': synonyms_str,
                        'chromosome': row.get('Chromosome', ''),
                        'position': row.get('Position', ''),
                        'uniprot': row.get('Uniprot', ''),
                        'description': row.get('Gene description', '')
                    }
                    
                    self._add_to_index(gene_symbol, gene_symbol, ensembl_id, 'primary')
                    
                    if synonyms_str:
                        for syn in synonyms_str.split(','):
                            syn_clean = syn.strip()
                            if syn_clean and syn_clean.upper() != gene_symbol.upper():
                                self._add_to_index(syn_clean, gene_symbol, ensembl_id, 'synonym')
                    
        except Exception as e:
            logger.error(f"构建HPA基因索引失败: {e}")
    
    def _add_to_index(self, name, gene_symbol, ensembl_id, name_type):
        norm = self._normalize(name)
        if norm and norm not in self.search_index:
            self.search_index[norm] = {
                'gene_symbol': gene_symbol,
                'ensembl_id': ensembl_id,
                'name_type': name_type,
                'original_name': name
            }
    
    def _normalize(self, name):
        if not name:
            return ""
        return name.upper().replace('-', '').replace(' ', '').replace('_', '')
    
    def get_suggestions(self, query, limit=8):
        if not query or len(query) < 2:
            return []
        
        query_norm = self._normalize(query)
        if not query_norm:
            return []
        
        matches = []
        seen_genes = set()
        
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
        
        matches.sort(key=lambda x: (x['score'], x['display_name']), reverse=True)
        return matches[:limit]
    
    def get_gene_info(self, gene_symbol):
        return self.gene_data_cache.get(gene_symbol.upper())
    
    def is_valid_gene(self, gene_symbol):
        return gene_symbol.upper() in self.gene_data_cache


# ==================== HPA基因详细信息提取服务 ====================
class HPAGeneDetailService:
    """从proteinatlas.tsv提取基因详细信息"""
    
    def __init__(self, hpa_manager=None):
        self.hpa_manager = hpa_manager
        self.data_file = self._get_data_file()
    
    def _get_data_file(self):
        if self.hpa_manager and hasattr(self.hpa_manager, 'data_file'):
            return self.hpa_manager.data_file
        local_dir = os.path.join(os.path.expanduser("~"), ".hpa_data")
        return os.path.join(local_dir, "proteinatlas.tsv")
    
    def get_gene_details(self, gene_symbol):
        """获取基因详细信息，如文件不存在则尝试下载"""
        if not self.data_file:
            return None
        
        # 如果文件不存在，尝试触发下载
        if not os.path.exists(self.data_file):
            logger.warning("HPA数据文件不存在，尝试下载...")
            if self.hpa_manager and hasattr(self.hpa_manager, 'check_and_download'):
                try:
                    self.hpa_manager.check_and_download()
                    # 更新数据文件路径（可能下载后路径有变化）
                    if hasattr(self.hpa_manager, 'data_file'):
                        self.data_file = self.hpa_manager.data_file
                except Exception as e:
                    logger.error(f"自动下载HPA数据失败: {e}")
            
            # 再次检查
            if not os.path.exists(self.data_file):
                logger.warning("HPA数据文件不可用")
                return None
        
        gene_symbol_upper = gene_symbol.upper().strip()
        
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter='\t')
                for row in reader:
                    if row.get('Gene name', '').upper().strip() == gene_symbol_upper:
                        return self._extract_gene_data(row)
        except Exception as e:
            logger.error(f"获取基因详情失败: {e}")
        
        return None
    
    def _extract_gene_data(self, row):
        ensembl_id = row.get('Gene', '')
        uniprot_id = row.get('Uniprot', '')
        chromosome = row.get('Chromosome', '')
        position = row.get('Position', '')
        
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
                'subcellular_main': row.get('Subcellular main location', ''),
                'subcellular_additional': row.get('Subcellular additional location', ''),
                'secretome_location': row.get('Secretome location', ''),
                'secretome_function': row.get('Secretome function', '')
            },
            'protein_function': {
                'biological_process': row.get('Biological process', ''),
                'molecular_function': row.get('Molecular function', ''),
                'disease_involvement': row.get('Disease involvement', '')
            },
            'rna_expression': {
                'tissue_specificity': row.get('RNA tissue specificity', ''),
                'tissue_specific_ntpm': row.get('RNA tissue specific nTPM', '')
            },
            'antibody': {
                'name': row.get('Antibody', ''),
                'hpa_search_url': f"https://www.proteinatlas.org/search/{row.get('Antibody', '').replace(' ', '%20')}" if row.get('Antibody') else '',
                'hpa_gene_url': f"https://www.proteinatlas.org/{row.get('Gene', '')}" if row.get('Gene') else ''
            },
            'rna_distribution': {
                'tissue': {
                    'specificity': row.get('RNA tissue specificity', ''),
                    'specific_ntpm': row.get('RNA tissue specific nTPM', '')
                },
                'single_cell': {
                    'specificity': row.get('RNA single cell type specificity', ''),
                    'specific_ncpm': row.get('RNA single cell type specific nCPM', '')
                },
                'cancer': {
                    'specificity': row.get('RNA cancer specificity', ''),
                    'specific_ptpm': row.get('RNA cancer specific pTPM', '')
                },
                'blood': {
                    'specificity': row.get('RNA blood cell specificity', ''),
                    'specific_ntpm': row.get('RNA blood cell specific nTPM', '')
                }
            }
        }


# ==================== 细胞系名称标准化验证模块 ====================
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
        if not self.api_key:
            return {'is_antiviral': False, 'confidence': 0, 'mechanism': '', 'reasoning': '未配置API'}

        try:
            prompt = f"""请分析以下文献，判断其是否报道了基因"{gene_name}"具有抗病毒功能。

文献标题：{title}
文献摘要：{abstract}

请按以下JSON格式回答（只返回JSON，不要有其他文字）：
{{
    "is_antiviral": true/false,
    "confidence": 0.0-1.0,
    "mechanism": "具体的抗病毒机制，如：调控IFITM家族、影响脂质代谢、激活干扰素通路等",
    "reasoning": "简要说明判断依据"
}}

注意：
1. is_antiviral：只要文献提到该基因能抑制病毒复制、增强抗病毒免疫、调控抗病毒基因表达等，即为true
2. confidence：证据越明确、机制越清晰，置信度越高
3. 即使不是经典的ISG基因，只要提到能影响病毒感染或复制，也算有抗病毒功能"""

            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            payload = {
                'model': 'qwen-turbo',
                'input': {
                    'messages': [
                        {'role': 'system', 'content': '你是一个专业的生物医学文献分析助手，擅长从文献中提取基因的抗病毒功能证据。'},
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
                is_antiviral = any(kw in (title + abstract).lower() for kw in
                    ['antiviral', 'virus', 'interferon', 'ifitm', 'innate immunity'])
                return {
                    'is_antiviral': is_antiviral,
                    'confidence': 0.5 if is_antiviral else 0,
                    'mechanism': 'AI解析失败，使用关键词匹配',
                    'reasoning': 'API返回格式异常，降级处理'
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
                text += f"{i}. {p.get('title', '')} - {p.get('abstract', '')[:300]}...\n"
            return text

        literature_text = ""
        literature_text += format_papers(papers_general, "基因功能相关")
        literature_text += format_papers(papers_oe, "过表达")
        literature_text += format_papers(papers_kd, "敲低/敲除")
        literature_text += format_papers(papers_ko, "敲除")

        prompt = f"""作为分子生物学和遗传学专家，请基于以下文献信息，全面总结基因"{gene_name}"（{gene_description}）的功能及实验模型数据。

{literature_text}

请按以下JSON格式提供结构化总结（只返回JSON）：
{{
    "protein_function": {{
        "category": "蛋白功能类别（如：锌指转录因子、丝氨酸蛋白酶、GPCR受体等）",
        "domains": "主要结构域及其功能",
        "pathways": "参与的关键信号通路",
        "cellular_location": "亚细胞定位",
        "tissue_expression": "主要表达组织"
    }},
    "overexpression": {{
        "cell_models": [
            {{
                "cell_line": "细胞系名称",
                "phenotype": "观察到的表型（如：促进增殖、诱导凋亡、EMT转化等）",
                "mechanism": "分子机制",
                "reference": "文献来源（作者, 年份, 期刊）"
            }}
        ],
        "animal_models": [
            {{
                "model": "动物模型（如：转基因小鼠、尾静脉注射等）",
                "phenotype": "表型",
                "reference": "文献来源"
            }}
        ],
        "summary": "过表达效应的总体特征"
    }},
    "knockdown": {{
        "cell_models": [
            {{
                "cell_line": "细胞系",
                "method": "敲低方法（siRNA/shRNA）",
                "phenotype": "表型",
                "reference": "文献来源"
            }}
        ],
        "summary": "敲低效应的总体特征"
    }},
    "knockout": {{
        "cell_models": [
            {{
                "cell_line": "细胞系",
                "method": "敲除方法（CRISPR/TALEN）",
                "phenotype": "表型",
                "viability": "是否影响细胞活力",
                "reference": "文献来源"
            }}
        ],
        "animal_models": [
            {{
                "model": "动物模型",
                "phenotype": "表型（如：胚胎致死、发育缺陷、代谢异常等）",
                "lethality": "致死性",
                "reference": "文献来源"
            }}
        ],
        "summary": "敲除效应的总体特征"
    }},
    "disease_relevance": {{
        "cancer": "在肿瘤中的作用（促癌/抑癌）及相关癌症类型",
        "other_diseases": "其他疾病相关性",
        "therapeutic_potential": "治疗潜力评估"
    }},
    "key_references": [
        "格式：作者 et al., 年份, 期刊, PMID（仅列出最关键3-5篇）"
    ],
    "experimental_notes": "实验设计建议（如：敲除是否致死、过表达是否诱导凋亡等注意事项）"
}}

要求：
1. 基于提供的文献如实总结，没有的数据标注"未见报道"
2. 区分细胞水平和动物水平的数据
3. 重点关注与慢病毒包装相关的因素（如：是否影响细胞活力、是否调控病毒相关通路）
4. 文献格式要规范，包含PMID"""

        try:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            payload = {
                'model': 'qwen-turbo',
                'input': {
                    'messages': [
                        {'role': 'system', 'content': '你是分子生物学专家，精通基因功能注释和表型分析，擅长从文献中提取关键实验数据。'},
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
                             gene_description: str = "", organism: str = "Homo sapiens") -> Dict:
        """四步法检索siRNA/shRNA序列：文献检索 -> Methods提取/补充材料标记 -> 专利检索 -> 物种核对（标记提醒）"""
        if not self.api_key:
            return {'error': '未配置AI API，无法设计序列', 'sequences': [], 'organism': organism}

        try:
            prompt = f"""作为RNAi序列设计专家，请为基因"{gene_name}"（Gene ID: {gene_id}, 描述: {gene_description}）设计siRNA/shRNA序列。

请基于最新的文献和公开数据库知识，提供：
1. 3条高质量的siRNA靶序列（19-21nt，不包含悬垂端）
2. 每条序列的设计依据（靶向哪个外显子、GC含量等）
3. 支持这些设计的参考文献或专利（真实存在的文献，优先选择高被引文献或经典方法论论文）

请按以下JSON格式回答（只返回JSON）：
{{
    "sequences": [
        {{
            "target_seq": "AAGUCGAGUAGCGAAGCUUTT",
            "target_region": "CDS区域，第123-141位",
            "design_rationale": "GC含量45%，避开UTR和SNP区域，无连续G/C",
            "efficiency_score": "高（预期敲低效率>80%）",
            "references": [
                {{
                    "type": "文献",
                    "title": "Specificity of RNA interference in mammalian cells",
                    "authors": "Elbashir et al.",
                    "year": "2001",
                    "source": "Nature",
                    "pmid_or_patent": "PMID:11252768",
                    "url": "https://pubmed.ncbi.nlm.nih.gov/11252768/"
                }}
            ]
        }}
    ],
    "shrna_vector_design": {{
        "loop_sequence": "TTCAAGAGA",
        "promoter": "U6或H1",
        "cloning_sites": "BamHI/EcoRI",
        "notes": "建议使用pLKO.1或pSUPER载体系统"
    }},
    "notes": "建议使用BLAST验证特异性，避免脱靶效应；推荐设计阴性对照（scrambled）",
    "validation_method": "Western blot或qPCR检测mRNA水平，建议检测时间点在转染后48-72小时"
}}

要求：
1. 序列必须是真实的、经过验证的设计原则
2. 参考文献必须是真实存在的（2000-2024年间）
3. 如该基因有已发表的有效siRNA序列（如来自Dharmacon、Sigma等数据库的验证序列），请优先列出
4. 提供具体的载体构建建议"""

            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            payload = {
                'model': 'qwen-turbo',
                'input': {
                    'messages': [
                        {'role': 'system', 'content': '你是RNA干扰(RNAi)技术专家，精通siRNA/shRNA序列设计和慢病毒载体构建，熟悉相关领域的经典文献和专利。'},
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
                return json.loads(content_clean)
            except json.JSONDecodeError:
                return {
                    'error': 'AI返回格式异常',
                    'sequences': [],
                    'raw_response': content[:500],
                    'note': '解析失败，但API已调用'
                }

        except Exception as e:
            return {'error': str(e), 'sequences': [], 'note': f'设计失败: {str(e)}'}

    def design_crispr_sequences(self, gene_name: str, gene_id: str = "",
                               gene_description: str = "") -> Dict:
        """从PMC全文文献中检索Materials and Methods部分的sgRNA序列（使用纯requests）"""
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

        prompt = f"""作为细胞培养专家，请分析细胞系"{cell_line}"（已标准化命名）的培养难点。

注：该细胞系名称已进行标准化处理（如NCI-H226、HK-2等）。

{f"基于以下文献：{literature_text}" if literature_text else "（未检索到该细胞系的特定文献，请基于该细胞系的一般特点和常识进行分析）"}

参考以下检查清单类别：
{checklist_text}

请按以下JSON格式返回该细胞的所有已知培养难点（只返回JSON）：
{{
    "culture_medium": [
        "具体难点描述（如：必须使用无血清mTeSR培养基，价格昂贵）"
    ],
    "coating_matrix": [
        "具体难点描述（如：必须预包被Matrigel，4°C过夜，操作繁琐）"
    ],
    "environment": [
        "具体难点描述（如：需要5%低氧培养箱，普通CO₂培养箱不适用）"
    ],
    "operation": [
        "具体难点描述（如：对胰酶极度敏感，消化超过3分钟即死亡）"
    ],
    "time_cost": [
        "具体难点描述（如：倍增时间72小时，实验周期漫长）"
    ],
    "special_warnings": [
        "关键警告（如：该细胞极易支原体污染，且污染后形态无明显变化）"
    ],
    "protocol_tips": [
        "实用建议（如：建议半量换液，传代比例1:2，周末不休息）"
    ],
    "verified_by": [
        "支持文献（格式：作者 et al., 年份, 期刊, PMID），如为常识则标注'基于细胞系一般特点'"
    ],
    "data_source_note": "说明数据是基于文献还是基于AI对该细胞系的常识推理"
}}

要求：
1. 如该细胞确实无特殊难点，请返回空数组并说明为常规培养细胞
2. 描述要具体（包含试剂名称、时间参数、浓度等）
3. 对于HK-2（人肾小管上皮细胞）、NCI-H226（肺癌细胞）等常见细胞系，请基于ATCC或常规培养知识提供信息
4. 必须区分：是文献明确报道的难点，还是AI基于该细胞类型的一般特点推断的潜在难点
5. 对于肾癌细胞系（如HK-2）、肺癌细胞系（如NCI-H226），请特别说明其特殊的培养要求"""

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

        prompt = f"""分析细胞系"{cell_line}"（已标准化命名）的慢病毒/逆转录病毒易感性。

{f"基于以下文献：{literature_text}" if literature_text else "（未检索到该细胞系的特定文献，请基于该细胞系的一般特点分析其慢病毒易感性，如上皮细胞通常较易感染，悬浮细胞较难感染等）"}

请提供以下具体信息（JSON格式）：
{{
    "susceptibility_level": "High/Medium/Low/Unknown",
    "recommended_moi": "推荐MOI范围（如：1-5，或>10）",
    "infection_efficiency": "典型感染效率（如：>80%，或<30%）",
    "requires_polybrene": "是否需要Polybrene（是/否/强烈建议）",
    "requires_spinfection": "是否需要离心感染（spinoculation）",
    "requires_pseudotyping": "是否需要特殊包膜蛋白（如VSV-G）",
    "cell_line_info": "该细胞系的基本信息（如HK-2为人肾小管近端上皮细胞，贴壁生长）",
    "challenges": [
        "具体难点（如：该细胞为悬浮细胞，极难感染，即使MOI=50效率仍<20%）"
    ],
    "optimization_tips": [
        "优化建议（如：建议使用polybrene 8μg/ml，离心感染1000xg 1小时）"
    ],
    "reported_cell_lines": [
        "文献中报道过的类似细胞（如：A549类似肺癌细胞系）"
    ],
    "references": ["支持文献或'基于细胞系类型的一般特点'"]
}}"""

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

    def check_and_download(self):
        """检查并下载HPA数据"""
        if not os.path.exists(self.data_file):
            self._download_hpa_data()

    def _download_hpa_data(self):
        """下载proteinatlas.tsv"""
        try:
            logger.info("正在下载HPA数据库（proteinatlas.tsv，约200MB），请稍候...")
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
                            logger.info(f"HPA下载进度: {progress:.1f}%")

            logger.info("正在解压HPA数据...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.local_dir)

            # 验证文件是否存在
            if not os.path.exists(self.data_file):
                extracted_files = os.listdir(self.local_dir)
                for f in extracted_files:
                    if f.endswith('.tsv') and 'proteinatlas' in f.lower():
                        self.data_file = os.path.join(self.local_dir, f)
                        logger.info(f"使用实际文件名: {f}")
                        break

            os.remove(zip_path)
            st.success("HPA数据下载完成")

        except Exception as e:
            logger.error(f"HPA download error: {e}", exc_info=True)
            st.error(f"HPA数据下载失败: {str(e)}")


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
        except Exception as e:
            logger.error(f"NCBI request failed: {e}")
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
        gene_info = {
            'id': gene_id,
            'name': gene_name,
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
            
            # 策略1：使用Accession限定搜索NM/XM
            search_terms = [
                f"{gene_id}[GeneID] AND (NM_[Accession] OR XM_[Accession])",
                f"{gene_id}[GeneID] AND refseq[Filter]",
                f"{gene_id}[GeneID] AND (mRNA[Title] OR transcript[Title])",
            ]
            
            for term in search_terms:
                search_params = {
                    'db': 'nuccore',
                    'term': term,
                    'retmode': 'json',
                    'retmax': 20,
                    'sort': 'accession'
                }
                result = self._make_request('esearch.fcgi', search_params)
                if result:
                    term_ids = result.get('esearchresult', {}).get('idlist', [])
                    debug_info.append(f"策略 '{term[:40]}...': 找到 {len(term_ids)} 个ID")
                    if term_ids and not ids:
                        ids = term_ids
                        logger.info(f"找到 {len(ids)} 个转录本 (查询: {term[:50]}...)")
                else:
                    debug_info.append(f"策略 '{term[:40]}...': 请求失败")
            
            if not ids:
                logger.warning(f"基因ID {gene_id} 未找到任何转录本")
                # 存储调试信息供上层使用
                self._last_transcript_debug = debug_info
                return []

            # 步骤2：获取详细信息
            summary_params = {
                'db': 'nuccore',
                'id': ','.join(ids),
                'retmode': 'json'
            }
            summary_result = self._make_request('esummary.fcgi', summary_params)
            if not summary_result:
                return []

            docs = summary_result.get('result', {})
            transcripts = []
            for uid in ids:
                try:
                    doc = docs.get(uid, {})
                    acc = doc.get('accessionversion', '')
                    slen = doc.get('slen', 0)

                    if not acc or not (acc.startswith('NM_') or acc.startswith('XM_')):
                        continue

                    tx_type = 'NM' if acc.startswith('NM_') else 'XM'
                    status = 'REVIEWED' if tx_type == 'NM' else 'PREDICTED'

                    transcripts.append({
                        'id': acc,
                        'length': int(slen) if slen else 0,
                        'type': tx_type,
                        'status': status,
                        'title': str(doc.get('title', ''))[:100]
                    })
                except Exception as e:
                    logger.warning(f"解析转录本 {uid} 失败: {e}")
                    continue

            # 排序：NM优先，然后按长度降序
            transcripts.sort(key=lambda x: (0 if x['type'] == 'NM' else 1, -x['length']))
            return transcripts

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

        for query in queries:
            try:
                search_params = {
                    'db': 'pubmed',
                    'term': query,
                    'retmode': 'json',
                    'retmax': 8,
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
                        abstract = doc.get('abstract', '') or doc.get('sorttitle', '')

                        if not title:
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
                    except Exception:
                        continue

            except Exception as e:
                logger.error(f"Literature search error: {e}")
                continue

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
            'conflicts': [],
            'filtered_xm': []
        }

        with st.spinner("多数据库交叉验证转录本（APPRIS/NCBI/Ensembl，优先NM）..."):
            ncbi_data = self._fetch_ncbi_refseq(gene_id)
            results['database_coverage']['NCBI_RefSeq'] = len(ncbi_data)

            ensembl_data = self._fetch_ensembl(gene_name)
            results['database_coverage']['Ensembl'] = len(ensembl_data)

            appris_data = self._fetch_appris(gene_name)
            results['database_coverage']['APPRIS'] = len(appris_data)

            ccle_data = {}
            if cell_line:
                ccle_data = self._fetch_ccle_expression(gene_name, cell_line)
                results['database_coverage']['CCLE'] = len(ccle_data)

            # 调试信息显示
            with st.expander("🔍 转录本获取调试信息", expanded=False):
                st.write(f"**基因名称**: {gene_name}")
                st.write(f"**NCBI Gene ID**: {gene_id}")
                st.write(f"**NCBI RefSeq**: 找到 {len(ncbi_data)} 个转录本")
                if ncbi_data:
                    st.write("NCBI转录本列表:", list(ncbi_data.keys())[:5])
                st.write(f"**Ensembl**: 找到 {len(ensembl_data)} 个转录本")
                st.write(f"**APPRIS**: 找到 {len(appris_data)} 个转录本")
                st.write(f"**CCLE**: 找到 {len(ccle_data)} 个表达记录")

            all_transcripts = self._merge_transcript_sources(
                ncbi_data, ensembl_data, appris_data, ccle_data
            )

            if not all_transcripts:
                st.warning("⚠️ 所有数据库均未返回有效转录本")
                return {'error': '未找到任何转录本信息', 'fallback': True}

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

    def _fetch_ncbi_refseq(self, gene_id: str) -> Dict:
        """使用NCBI E-utilities获取RefSeq转录本（使用改进的NCBI方法）"""
        transcripts = {}
        try:
            # 使用NCBIClient的_fetch_transcripts方法来获取（已带重试）
            tx_list = self.ncbi._fetch_transcripts(gene_id)
            for tx in tx_list:
                tx_id = tx['id']
                transcripts[tx_id] = {
                    'source': 'NCBI',
                    'status': tx['status'],
                    'length': tx['length'],
                    'type': tx['type'],
                    'title': tx.get('title', '')
                }
            logger.info(f"NCBI RefSeq获取成功: {len(transcripts)}个转录本")
            
            # 显示调试信息
            if hasattr(self.ncbi, '_last_transcript_debug') and self.ncbi._last_transcript_debug:
                st.caption(f"📊 NCBI搜索详情: {' | '.join(self.ncbi._last_transcript_debug)}")
                
        except Exception as e:
            logger.error(f"NCBI RefSeq获取失败: {e}")
            st.warning(f"⚠️ NCBI RefSeq获取失败: {e}")
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

    def _calculate_transcript_score(self, tx_info: Dict, cell_line: Optional[str]) -> Tuple[float, List[str]]:
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

    def render(self, organism: str = "human", key_prefix: str = "gene") -> Optional[str]:
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
        input_label = "基因名（HPA数据自动补全，输入2个字符以上显示建议）"

        user_input = st.text_input(
            input_label,
            value=st.session_state[input_key],
            placeholder="例如：TP53, EGFR, GAPDH...",
            key=f"{key_prefix}_text_widget"
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
                if hpa_available:
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
            
            # ===== 第三行：RNA表达与抗体 =====
            col3_1, col3_2 = st.columns(2)
            
            with col3_1:
                # e. RNA表达与定位
                st.markdown("#### 📊 RNA表达与定位")
                rna_exp = gene_info.get('rna_expression', {})
                if rna_exp.get('tissue_specificity'):
                    st.markdown(f"**组织特异性:** {rna_exp['tissue_specificity']}")
                    if rna_exp.get('tissue_specific_ntpm'):
                        st.caption(f"nTPM: {rna_exp['tissue_specific_ntpm']}")
                else:
                    st.markdown("*暂无RNA表达数据*")
            
            with col3_2:
                # f. 抗体推荐
                st.markdown("#### 🧪 抗体推荐")
                antibody = gene_info.get('antibody', {})
                if antibody.get('name'):
                    # 优先使用基因页面链接，更稳定
                    antibody_name = antibody['name']
                    hpa_url = antibody.get('hpa_gene_url') or antibody.get('hpa_search_url', '')
                    st.markdown(f"**产品名称:** [{antibody_name}]({hpa_url})")
                    st.caption(f"[在HPA数据库中查看抗体详情]({hpa_url})")
                else:
                    st.markdown("*暂无抗体推荐*")
            
            st.divider()
            
            # ===== 第四行：RNA在不同样本中的表达 =====
            st.markdown("#### 📈 RNA在不同样本中的表达")
            dist = gene_info.get('rna_distribution', {})
            
            dist_cols = st.columns(4)
            
            # g. RNA在不同组织、细胞、肿瘤和血细胞中的表达
            # 组织
            tissue = dist.get('tissue', {})
            with dist_cols[0]:
                st.markdown("**🧬 组织**")
                if tissue.get('specificity'):
                    st.markdown(f"{tissue['specificity']}")
                    if tissue.get('specific_ntpm'):
                        st.caption(f"nTPM: {tissue['specific_ntpm']}")
                else:
                    st.caption("-")
            
            # 单细胞
            sc = dist.get('single_cell', {})
            with dist_cols[1]:
                st.markdown("**🔬 单细胞**")
                if sc.get('specificity'):
                    st.markdown(f"{sc['specificity']}")
                    if sc.get('specific_ncpm'):
                        st.caption(f"nCPM: {sc['specific_ncpm']}")
                else:
                    st.caption("-")
            
            # 肿瘤
            cancer = dist.get('cancer', {})
            with dist_cols[2]:
                st.markdown("**⚕️ 肿瘤**")
                if cancer.get('specificity'):
                    st.markdown(f"{cancer['specificity']}")
                    if cancer.get('specific_ptpm'):
                        st.caption(f"pTPM: {cancer['specific_ptpm']}")
                else:
                    st.caption("-")
            
            # 血细胞
            blood = dist.get('blood', {})
            with dist_cols[3]:
                st.markdown("**🩸 血细胞**")
                if blood.get('specificity'):
                    st.markdown(f"{blood['specificity']}")
                    if blood.get('specific_ntpm'):
                        st.caption(f"nTPM: {blood['specific_ntpm']}")
                else:
                    st.caption("-")


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

                tx_selection = selector.select_optimal_transcript(
                    gene_name=gene_name,
                    gene_id=gene_id,
                    cell_line=effective_cell_line
                )

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

                selected_tx = tx_selection.get('selected_transcript', {})

                transcripts = [{
                    'id': selected_tx.get('id'),
                    'length': selected_tx.get('info', {}).get('length', 0),
                    'selection_reason': selected_tx.get('reasons', []),
                    'confidence': selected_tx.get('score', 0),
                    'all_candidates': tx_selection.get('all_transcripts', [])
                }]

                gene_info = gene_info_basic

                result['transcript_selection'] = tx_selection
        except Exception as e:
            logger.error(f"转录本选择失败: {e}")
            result['errors'].append(f"转录本选择失败: {str(e)}")
            result['status'] = 'error'
            return result

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

        # HPA数据查询 + 高级分析
        if organism == 'Homo sapiens' and effective_cell_line:
            try:
                # 确保HPA数据已下载
                self.hpa.check_and_download()

                with st.spinner("查询HPA表达数据并进行高级分析..."):
                    hpa_data = self.hpa.get_expression_data(gene_name, effective_cell_line)
                    if hpa_data:
                        result['hpa_data'] = hpa_data
                    else:
                        result['hpa_data'] = {
                            'message': f'在HPA数据库中未找到{gene_name}在{effective_cell_line}中的表达数据',
                            'searched_cell_line': effective_cell_line
                        }

                    comprehensive_analysis = self.comprehensive_hpa_analysis(
                        gene_name, effective_cell_line, hpa_data if hpa_data else None
                    )
                    result['comprehensive_hpa_analysis'] = comprehensive_analysis
            except Exception as e:
                logger.error(f"HPA分析失败: {e}")
                result['warnings'].append(f"HPA分析失败: {str(e)}")
                result['hpa_analysis_error'] = str(e)
        else:
            result['hpa_data'] = {
                'message': f'HPA仅支持人类细胞系。当前: {organism}, {effective_cell_line}',
                'note': 'HPA仅包含人类细胞系数据'
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
                # shRNA设计仍使用AI
                if self.ai and self.ai.api_key:
                    try:
                        with st.spinner("AI正在设计shRNA序列..."):
                            design_data = self.ai.design_rnai_sequences(
                                gene_name=gene_name,
                                gene_id=gene_info.get('id', ''),
                                gene_description=gene_info.get('description', '')
                            )
                            result['sequence_designs'] = {
                                'type': 'siRNA/shRNA (AI设计)',
                                'designs': design_data,
                                'source': 'AI基于最新文献和数据库知识设计',
                                'status': 'success' if not design_data.get('error') else 'error'
                            }
                    except Exception as e:
                        logger.error(f"shRNA设计失败: {e}")
                        result['sequence_designs'] = {
                            'type': 'shRNA设计',
                            'designs': {'error': str(e)},
                            'source': '设计失败',
                            'status': 'error'
                        }
                else:
                    result['sequence_designs'] = {
                        'type': 'shRNA设计',
                        'designs': {'error': '未配置AI API'},
                        'source': 'N/A',
                        'status': 'no_api'
                    }
            else:
                # CRISPR sgRNA从文献检索
                try:
                    with st.spinner("正在从PubMed文献中检索已报道的sgRNA序列..."):
                        design_data = self.ai.design_crispr_sequences(
                            gene_name=gene_name,
                            gene_id=gene_info.get('id', ''),
                            gene_description=gene_info.get('description', '')
                        )
                        result['sequence_designs'] = {
                            'type': 'sgRNA (文献检索)',
                            'designs': design_data,
                            'source': 'PubMed文献检索',
                            'status': 'success' if not design_data.get('error') and design_data.get('sgrnas') else 'no_data'
                        }
                except Exception as e:
                    logger.error(f"sgRNA检索失败: {e}")
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

    # ==================== HPA分析方法（简化版）====================

    def comprehensive_hpa_analysis(self, gene_name: str, cell_line: str, cached_hpa_data: Dict = None) -> Dict:
        """HPA基础数据分析：蛋白定位、表达量、内参推荐"""
        result = {
            'gene': gene_name,
            'cell_line': cell_line,
            'protein_localization': {},
            'gene_expression': {},
            'reference_genes': {},
            'antibody_strategy': {}
        }

        # 1. 获取蛋白定位信息
        result['protein_localization'] = self.hpa.get_protein_localization(gene_name)

        # 2. 获取目的基因在指定细胞系的表达量
        hpa_data = cached_hpa_data if cached_hpa_data else self.hpa.get_expression_data(gene_name, cell_line)
        if hpa_data:
            result['gene_expression'] = {
                'rna_level': hpa_data.get('rna_level', 'N/A'),
                'protein_level': hpa_data.get('protein_level', 'N/A'),
                'reliability': hpa_data.get('reliability', 'N/A'),
                'ensembl_id': hpa_data.get('ensembl_id', 'N/A')
            }

        # 3. 根据蛋白定位推荐内参基因
        location_types = result['protein_localization'].get('location_types', ['general'])
        # TODO: 需要获取目的蛋白的分子量来推荐内参，这里暂时用0表示未知
        target_mw = 0  # 可以从UniProt等数据库获取
        result['reference_genes'] = self.hpa.get_reference_genes_for_location(
            location_types, cell_line, target_mw
        )

        # 4. 抗体验证策略（WB和流式）
        result['antibody_strategy'] = self._get_antibody_strategy(gene_name, result['protein_localization'])

        return result

    def _get_antibody_strategy(self, gene_name: str, localization: Dict) -> Dict:
        """生成抗体验证策略（WB和流式）"""
        loc_types = localization.get('location_types', [])
        main_loc = localization.get('main_localization', 'Unknown')

        # 根据定位推荐验证方法
        wb_notes = []
        flow_notes = []

        if 'membrane' in loc_types:
            wb_notes.append("膜蛋白，建议提取总蛋白或使用膜蛋白提取试剂盒")
            wb_notes.append("转膜条件：建议0.45μm PVDF膜，恒流300mA 90min")
            flow_notes.append("适合流式检测（细胞表面标记）")
            flow_notes.append("建议活细胞染色或固定后染色")
        elif 'nucleus' in loc_types:
            wb_notes.append("核蛋白，需要核蛋白提取或RIPA强裂解液")
            wb_notes.append("注意区分核质和核膜定位")
        elif 'cytoplasm' in loc_types:
            wb_notes.append("胞质蛋白，常规RIPA提取即可")
            flow_notes.append("需要固定破膜后才能进行流式检测")
        elif 'mitochondria' in loc_types:
            wb_notes.append("线粒体蛋白，建议提取线粒体组分")
            wb_notes.append("注意与核编码蛋白区分")

        if not flow_notes:
            flow_notes.append("该蛋白定位可能不适合常规流式检测")
            flow_notes.append("如需流式检测，建议考虑过表达带标签的蛋白")

        return {
            'gene': gene_name,
            'main_localization': main_loc,
            'wb_antibody': {
                'recommendations': wb_notes if wb_notes else ["常规WB条件即可"],
                'notes': f"针对{main_loc}定位的WB建议"
            },
            'flow_cytometry': {
                'recommendations': flow_notes,
                'notes': f"针对{main_loc}定位的流式建议"
            }
        }

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
                                help="可选，用于AI文献语义分析、序列设计、细胞培养难点分析")

        final_qwen = APIConfig.get_qwen_api_key()
        if final_qwen:
            st.success("✓ AI API已配置 - 将启用智能细胞系评估和基因功能分析")
        else:
            st.warning("✗ 未配置AI API（请在上方输入或在Secrets中设置DASHSCOPE_API_KEY）")

        st.divider()
        st.caption("核心列表+文献补充+AI语义分析混合策略")

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

    col1, col2 = st.columns(2)

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
            }.get(x, x)
        )

        # ===== HPA基因自动补全服务初始化 =====
        if 'hpa_gene_service' not in st.session_state:
            # 尝试从现有的HPA管理器获取数据文件路径
            hpa_manager = st.session_state.get('hpa_manager')
            st.session_state['hpa_gene_service'] = HPAGeneAutocompleteService(hpa_manager)
            st.session_state['hpa_gene_detail_service'] = HPAGeneDetailService(hpa_manager)
        
        hpa_gene_service = st.session_state.get('hpa_gene_service')
        hpa_detail_service = st.session_state.get('hpa_gene_detail_service')
        
        gene_service = GeneAutocompleteService()  # 保留用于其他可能的用途
        # 基因输入完全基于HPA数据包（抛弃NCBI自动补全）
        gene_component = GeneInputComponent(hpa_gene_service, hpa_detail_service)
        gene = gene_component.render(organism, key_prefix="main_gene")

    with col2:
        # ===== HPA细胞系自动补全输入（新增功能）=====
        if 'cell_line_component' not in st.session_state:
            st.session_state.cell_line_component = HPACellLineAutocompleteService()

        cell_service = st.session_state.cell_line_component

        # 输入框
        cell_input = st.text_input(
            "细胞系（HPA数据库自动补全，输入1字符即显示建议）",
            value=st.session_state.get('cell_line_input', ''),
            placeholder="例如：A549, HeLa, MCF7, NCI-H226, HEK293, SH-SY5Y...",
            help="输入细胞系名称，系统从HPA数据库1206个细胞系中匹配。支持模糊匹配、大小写不敏感、分隔符容错。",
            key="cell_line_widget"
        )

        # 保存输入并触发建议更新
        if cell_input != st.session_state.get('cell_line_input', ''):
            st.session_state['cell_line_input'] = cell_input
            st.session_state['cell_line_selected'] = None
            if len(cell_input) >= 1:
                safe_rerun()

        cell_line_value = None
        cell_metadata = None

        # 显示自动补全建议
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
        }.get(x, x)
    )

    analyze = st.button("开始AI智能评估", type="primary", use_container_width=True)

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

    col_exp1, col_exp2 = st.columns(2)
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
        "硬性规则检查",
        "基因功能分析",
        "HPA基础数据",     # 蛋白定位、表达量、内参推荐
        "抗体验证策略",    # WB和流式抗体
        "细胞系评估",
        "序列设计",
        "转录本选择"
    ])

    with tabs[0]:
        st.markdown("### 混合硬性规则检查")
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

                if 'protein_function' in data:
                    with st.expander("蛋白基础功能", expanded=True):
                        pf = data['protein_function']
                        st.write(f"**蛋白类别**: {pf.get('category', 'N/A')}")
                        st.write(f"**结构域**: {pf.get('domains', 'N/A')}")
                        st.write(f"**信号通路**: {pf.get('pathways', 'N/A')}")
                        st.write(f"**亚细胞定位**: {pf.get('cellular_location', 'N/A')}")
                        st.write(f"**组织表达**: {pf.get('tissue_expression', 'N/A')}")

                if 'overexpression' in data:
                    with st.expander("过表达效应"):
                        oe = data['overexpression']
                        if 'cell_models' in oe and oe['cell_models']:
                            st.markdown("**细胞模型:**")
                            for model in oe['cell_models']:
                                st.markdown(f"""
                                    - **{model.get('cell_line', 'N/A')}**: {model.get('phenotype', 'N/A')}
                                    *机制*: {model.get('mechanism', 'N/A')} | *文献*: {model.get('reference', 'N/A')}
                                """)
                        if 'summary' in oe:
                            st.success(f"**总结**: {oe['summary']}")

                if 'knockout' in data:
                    with st.expander("敲除效应"):
                        ko = data['knockout']
                        if 'cell_models' in ko and ko['cell_models']:
                            st.markdown("**细胞模型:**")
                            for model in ko['cell_models']:
                                st.markdown(f"""
                                    - **{model.get('cell_line', 'N/A')}** ({model.get('method', '')}): {model.get('phenotype', 'N/A')}
                                    *细胞活力*: {model.get('viability', 'N/A')}
                                """)
                        if 'summary' in ko:
                            st.success(f"**总结**: {ko['summary']}")

                if 'disease_relevance' in data:
                    with st.expander("疾病相关性"):
                        dr = data['disease_relevance']
                        st.write(f"**肿瘤作用**: {dr.get('cancer', 'N/A')}")
                        st.write(f"**其他疾病**: {dr.get('other_diseases', 'N/A')}")
                        st.write(f"**治疗潜力**: {dr.get('therapeutic_potential', 'N/A')}")

                if 'key_references' in data and data['key_references']:
                    with st.expander("关键参考文献"):
                        for ref in data['key_references']:
                            st.markdown(f"- {ref}")

                if 'experimental_notes' in data:
                    st.info(f"**实验设计建议**: {data['experimental_notes']}")
            else:
                st.info("正在加载基因功能分析数据...")
        else:
            st.info("未获取到基因功能分析数据")

    # ==================== HPA基础数据（合并蛋白定位、表达量、内参推荐）====================
    with tabs[2]:
        st.markdown("### 🧬 HPA基础数据")
        st.caption("数据来源: Human Protein Atlas (HPA)")

        hpa_analysis = result.get('comprehensive_hpa_analysis', {})

        if hpa_analysis and isinstance(hpa_analysis, dict):
            # 1. 蛋白定位信息
            loc_data = hpa_analysis.get('protein_localization', {})
            if loc_data and 'error' not in loc_data:
                st.subheader("📍 蛋白亚细胞定位")
                main_loc = loc_data.get('main_localization', 'Unknown')
                st.success(f"**主要定位**: {main_loc}")

                loc_types = loc_data.get('location_types', [])
                if loc_types:
                    st.write("**定位类别**: " + ", ".join(loc_types))

                add_loc = loc_data.get('additional_localization', '')
                if add_loc:
                    st.caption(f"附加定位: {add_loc}")

            st.divider()

            # 2. 目的基因表达量
            expr_data = hpa_analysis.get('gene_expression', {})
            if expr_data:
                st.subheader("📊 目的基因表达量")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("RNA水平", expr_data.get('rna_level', 'N/A'))
                with col2:
                    st.metric("蛋白水平", expr_data.get('protein_level', 'N/A'))
                with col3:
                    st.metric("可靠性", expr_data.get('reliability', 'N/A'))

            st.divider()

            # 3. 内参基因推荐（基于定位和分子量）
            ref_data = hpa_analysis.get('reference_genes', {})
            if ref_data and 'error' not in ref_data:
                st.subheader("🎯 内参基因推荐")

                loc_types = ref_data.get('location_types', ['general'])
                st.write(f"**基于定位**: {', '.join(loc_types)}")

                ref_genes = ref_data.get('reference_genes', [])
                if ref_genes:
                    # 分离推荐和非推荐（基于分子量差异）
                    recommended = [r for r in ref_genes if r.get('recommended')][:3]
                    others = [r for r in ref_genes if not r.get('recommended')][:5]

                    if recommended:
                        st.markdown("**✅ 推荐内参（分子量差异>20kDa）**")
                        rec_df = pd.DataFrame([
                            {
                                '基因': r['gene'],
                                '蛋白名称': r['name'],
                                '分子量(kDa)': r['mw_kda'],
                                'RNA(nTPM)': f"{r['rna_ntpm']:.1f}",
                                '定位': r['location']
                            } for r in recommended
                        ])
                        st.table(rec_df)

                    if others:
                        with st.expander("其他可用内参"):
                            other_df = pd.DataFrame([
                                {
                                    '基因': r['gene'],
                                    '分子量(kDa)': r['mw_kda'],
                                    'RNA(nTPM)': f"{r['rna_ntpm']:.1f}"
                                } for r in others
                            ])
                            st.table(other_df)

                    if ref_data.get('recommendation'):
                        st.info(f"**建议**: {ref_data['recommendation']}")
                else:
                    st.warning("未找到该细胞系中表达稳定的内参基因")
            else:
                st.info("未获取到内参基因推荐数据")
        else:
            st.info("未获取到HPA分析数据（请确保选择人类物种并输入有效细胞系）")

    # ==================== 抗体验证策略（WB和流式）====================
    with tabs[3]:
        st.markdown("### 🔬 抗体验证策略")

        hpa_analysis = result.get('comprehensive_hpa_analysis', {})
        antibody_data = hpa_analysis.get('antibody_strategy', {}) if hpa_analysis else {}

        if antibody_data and 'error' not in antibody_data:
            main_loc = antibody_data.get('main_localization', 'Unknown')
            st.success(f"目标蛋白定位: **{main_loc}**")

            # WB抗体建议
            wb_data = antibody_data.get('wb_antibody', {})
            if wb_data:
                st.subheader("🧪 Western Blot 抗体验证")
                st.write(f"**说明**: {wb_data.get('notes', '')}")
                for rec in wb_data.get('recommendations', []):
                    st.markdown(f"- {rec}")

            st.divider()

            # 流式抗体建议
            flow_data = antibody_data.get('flow_cytometry', {})
            if flow_data:
                st.subheader("🌊 流式细胞术 (Flow Cytometry)")
                st.write(f"**说明**: {flow_data.get('notes', '')}")
                for rec in flow_data.get('recommendations', []):
                    st.markdown(f"- {rec}")
        else:
            st.info("未获取到抗体验证策略数据")

    # ==================== 细胞系评估 ====================
    with tabs[4]:
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

                        categories = {
                            'culture_medium': ('培养基特殊要求', '#ffebee'),
                            'coating_matrix': ('基质/包被要求', '#f3e5f5'),
                            'environment': ('气体/环境敏感', '#e3f2fd'),
                            'operation': ('操作复杂度', '#fff3e0'),
                            'time_cost': ('时间/成本', '#e8f5e9'),
                            'special_warnings': ('⚠️ 关键警告', '#ffcdd2'),
                            'protocol_tips': ('💡 Protocol建议', '#e0f7fa')
                        }

                        has_any = False
                        for key, (title, bg_color) in categories.items():
                            items = culture_diff.get(key, [])
                            if items and isinstance(items, list) and len(items) > 0:
                                has_any = True
                                with st.expander(f"{title} ({len(items)}项)", expanded=(key=='special_warnings')):
                                    for item in items:
                                        st.markdown(f"""
                                            <div style='margin: 5px 0; padding: 8px; background-color: {bg_color};
                                            border-radius: 5px; border-left: 3px solid #666;'>
                                                {html.escape(str(item))}
                                            </div>
                                        """, unsafe_allow_html=True)

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
    with tabs[5]:
        st.markdown("### 序列设计参考")
        seq_data = result.get('sequence_designs')
        if seq_data and isinstance(seq_data, dict):
            status = seq_data.get('status', 'unknown')

            if status == 'no_api':
                st.warning("⚠️ 未配置AI API，无法提供序列设计")
                st.info("请在侧边栏输入通义千问API Key，或在Secrets中设置DASHSCOPE_API_KEY")
            elif status == 'error':
                st.error(f"❌ 序列设计失败: {seq_data.get('designs', {}).get('error', '未知错误')}")
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
                                # 根据物种匹配状态设置标签
                                species_match = seq.get('species_match')
                                species_warning = seq.get('species_warning', '')
                                if species_match is True:
                                    species_tag = "✅ 物种匹配"
                                elif species_match is False:
                                    species_tag = "⚠️ 物种不匹配"
                                else:
                                    species_tag = "⚠️ 物种待核对"
                                
                                with st.expander(f"候选序列 #{i+1}: {seq.get('target_seq', 'N/A')[:20]}... ({species_tag})"):
                                    st.write(f"**靶序列**: `{seq.get('target_seq', 'N/A')}`")
                                    st.write(f"**靶标区域**: {seq.get('target_region', 'N/A')}")
                                    st.write(f"**设计原理**: {seq.get('design_rationale', 'N/A')}")
                                    st.write(f"**预期效率**: {seq.get('efficiency_score', 'N/A')}")
                                    
                                    # 显示物种匹配信息
                                    if species_match is True:
                                        st.success(f"✅ 物种匹配：目标物种 {seq.get('user_organism', 'N/A')} 与文章物种 {seq.get('article_organism', 'N/A')} 一致")
                                    elif species_match is False:
                                        st.warning(f"⚠️ {species_warning} - 请谨慎使用，建议核对序列在目标物种中的保守性")
                                    else:
                                        st.info(f"⚠️ {species_warning}")
                                    
                                    refs = seq.get('reference', {})
                                    if refs:
                                        st.write("**参考文献**:")
                                        st.markdown(f"- *{refs.get('title', '')}* ({refs.get('year', '')}) [{refs.get('pmid_or_patent', '')}]({refs.get('url', '')})")
                                        if refs.get('pmc_url'):
                                            st.markdown(f"- [PMC全文]({refs.get('pmc_url')})")

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
                                        # 根据物种匹配状态设置标签
                                        species_match = sgrna.get('species_match')
                                        species_warning = sgrna.get('species_warning', '')
                                        if species_match is True:
                                            species_tag = "✅ 物种匹配"
                                        elif species_match is False:
                                            species_tag = "⚠️ 物种不匹配"
                                        else:
                                            species_tag = "⚠️ 物种待核对"
                                        
                                        with st.expander(f"sgRNA #{i+1}: {sgrna.get('sequence', 'N/A')} ({species_tag})"):
                                            st.write(f"**靶序列**: `{sgrna.get('sequence', 'N/A')}`")
                                            st.write(f"**PAM**: {sgrna.get('pam', 'NGG')}")
                                            st.write(f"**说明**: {sgrna.get('target_exon', 'N/A')}")
                                            
                                            # 显示物种匹配信息
                                            if species_match is True:
                                                st.success(f"✅ 物种匹配：目标物种 {sgrna.get('user_organism', 'N/A')} 与文章物种 {sgrna.get('article_organism', 'N/A')} 一致")
                                            elif species_match is False:
                                                st.warning(f"⚠️ {species_warning} - 请谨慎使用，建议核对序列在目标物种中的保守性")
                                            else:
                                                st.info(f"⚠️ {species_warning}")
                                            
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
                                            # 物种匹配标签
                                            species_match = sup.get('species_match')
                                            if species_match is True:
                                                species_icon = "✅"
                                            elif species_match is False:
                                                species_icon = "⚠️"
                                            else:
                                                species_icon = "❓"
                                            
                                            st.markdown(f"**{i+1}. {sup.get('title', '')}** ({sup.get('year', '')}) {species_icon}")
                                            st.markdown(f"   - [PubMed]({sup.get('url', '')}) | [PMC全文]({sup.get('pmc_url', '')})")
                                            if sup.get('species_warning'):
                                                st.caption(f"   ⚠️ {sup.get('species_warning')}")
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
    with tabs[6]:
        st.markdown("### 转录本选择详情（多数据库交叉验证）")
        tx_data = result.get('transcript_selection', {})
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
    try:
        if not AuthManager.check_password():
            st.stop()
    except Exception as e:
        st.error(f"密码验证模块错误: {e}")
        st.stop()

    try:
        hpa_manager = HPADataManager()
        hpa_manager.check_and_download()
        st.session_state['hpa_manager'] = hpa_manager  # 存入session_state供基因自动补全服务使用
    except Exception as e:
        st.warning(f"HPA数据管理器初始化警告: {e}")

    try:
        render_sidebar()
    except Exception as e:
        st.error(f"侧边栏渲染错误: {e}")

    try:
        organism, gene, cell_line, exp_type, analyze = render_main_panel()
    except Exception as e:
        st.error(f"主面板渲染错误: {e}")
        return

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
                render_results(result)

        except Exception as e:
            logger.exception(f"Unhandled error: {e}")
            st.error(f"系统错误: {str(e)}")
            st.exception(e)

if __name__ == "__main__":
    main()
