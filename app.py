"""
慢病毒包装与细胞系构建评估系统
部署版本 - Streamlit Cloud
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
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from io import BytesIO, StringIO
import pandas as pd
import zipfile

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("lentivirus_assessment")

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="慢病毒包装与细胞系构建评估系统",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== AI分析客户端 ====================
class AIAnalysisClient:
    """通义千问API客户端 - 用于文献语义分析和序列设计"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
    
    def analyze_antiviral_evidence(self, gene_name: str, title: str, abstract: str) -> Dict:
        """
        使用AI分析文献是否包含抗病毒功能证据
        返回: {
            'is_antiviral': bool,
            'confidence': float,  # 0-1
            'mechanism': str,     # 机制描述
            'reasoning': str      # 推理过程
        }
        """
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
    "mechanism": "具体的抗病毒机制，如：调控IFITM家族、影响鞘脂代谢、激活干扰素通路等",
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
            
            # 解析JSON响应
            try:
                # 清理可能的Markdown代码块标记
                content_clean = content.replace('```json', '').replace('```', '').strip()
                analysis = json.loads(content_clean)
                
                return {
                    'is_antiviral': analysis.get('is_antiviral', False),
                    'confidence': float(analysis.get('confidence', 0)),
                    'mechanism': analysis.get('mechanism', ''),
                    'reasoning': analysis.get('reasoning', '')
                }
            except json.JSONDecodeError:
                logger.warning(f"AI返回非JSON格式: {content}")
                # 简单关键词回退
                is_antiviral = any(kw in (title + abstract).lower() for kw in 
                                  ['antiviral', 'virus', 'interferon', 'ifitm', 'innate immunity'])
                return {
                    'is_antiviral': is_antiviral,
                    'confidence': 0.5 if is_antiviral else 0,
                    'mechanism': 'AI解析失败，使用关键词匹配',
                    'reasoning': 'API返回格式异常，降级处理'
                }
                
        except Exception as e:
            logger.error(f"AI分析失败: {e}")
            return {
                'is_antiviral': False,
                'confidence': 0,
                'mechanism': '',
                'reasoning': f'API调用失败: {str(e)}'
            }
    
    def analyze_gene_function_comprehensive(self, gene_name: str, gene_description: str, 
                                           papers_oe: List[Dict], papers_kd: List[Dict], 
                                           papers_ko: List[Dict], papers_general: List[Dict]) -> Dict:
        """
        综合分析基因功能及不同实验模型的表型
        
        返回: {
            'protein_function': {
                'category': '蛋白类别（转录因子/酶/受体等）',
                'domains': '结构域',
                'pathways': '涉及的信号通路',
                'cellular_location': '亚细胞定位'
            },
            'overexpression': {
                'cell_models': [{'cell_line': '细胞系', 'phenotype': '表型', 'reference': '文献'}],
                'animal_models': [{'model': '动物模型', 'phenotype': '表型', 'reference': '文献'}],
                'summary': '过表达效应总结'
            },
            'knockdown': {
                'cell_models': [...],
                'summary': '敲低效应总结'
            },
            'knockout': {
                'cell_models': [...],
                'animal_models': [...],
                'summary': '敲除效应总结'
            },
            'disease_relevance': '疾病相关性',
            'key_references': ['关键文献列表']
        }
        """
        if not self.api_key:
            return {'error': '未配置AI API'}
        
        # 构建文献摘要文本
        def format_papers(papers, label):
            text = f"\n{label}文献：\n"
            for i, p in enumerate(papers[:5], 1):
                text += f"{i}. {p.get('title', '')} - {p.get('abstract', '')[:300]}...\n"
            return text
        
        literature_text = ""
        if papers_general:
            literature_text += format_papers(papers_general, "基因功能相关")
        if papers_oe:
            literature_text += format_papers(papers_oe, "过表达")
        if papers_kd:
            literature_text += format_papers(papers_kd, "敲低/敲除")
        if papers_ko:
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
            
            # 解析JSON
            try:
                content_clean = content.replace('```json', '').replace('```', '').strip()
                analysis = json.loads(content_clean)
                return analysis
            except json.JSONDecodeError:
                logger.error(f"AI功能分析返回非JSON格式: {content}")
                return {
                    'error': 'AI返回格式异常',
                    'raw_response': content[:1000]
                }
                
        except Exception as e:
            logger.error(f"AI功能分析失败: {e}")
            return {'error': str(e)}
    
    def design_rnai_sequences(self, gene_name: str, gene_id: str = "", gene_description: str = "") -> Dict:
        """
        使用AI设计siRNA/shRNA序列，并返回推荐序列及来源文献/专利
        
        返回: {
            'sequences': [
                {
                    'target_seq': '正义链序列(19-21nt)',
                    'design_rationale': '设计原理（靶向区域、GC含量等）',
                    'efficiency_score': '预期效率评分',
                    'references': [
                        {
                            'type': '文献/专利',
                            'title': '标题',
                            'authors': '作者/发明人',
                            'year': '年份',
                            'source': '期刊/专利号',
                            'pmid_or_patent': 'PMID或专利号',
                            'url': '链接'
                        }
                    ]
                }
            ],
            'notes': '设计注意事项',
            'validation_method': '推荐的验证方法'
        }
        """
        if not self.api_key:
            return {'error': '未配置AI API', 'sequences': []}
        
        try:
            prompt = f"""作为RNAi序列设计专家，请为基因"{gene_name}"（Gene ID: {gene_id}, 描述: {gene_description}）设计siRNA/shRNA序列。

请基于最新的文献和公开数据库知识，提供：
1. 3条高质量的siRNA靶序列（19-21nt，不包含悬垂端）
2. 每条序列的设计依据（靶向哪个外显子、GC含量、特异性考虑）
3. 支持这些设计的参考文献或专利（真实存在的文献，优先选择高被引文献或经典方法论文）

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
2. 参考文献必须是真实存在的经典文献（2000-2024年间）
3. 如果基因有已发表的有效siRNA序列（如来自Dharmacon、Sigma等数据库的验证序列），请优先列出
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
            
            # 解析JSON
            try:
                content_clean = content.replace('```json', '').replace('```', '').strip()
                design_data = json.loads(content_clean)
                return design_data
            except json.JSONDecodeError:
                logger.error(f"AI序列设计返回非JSON格式: {content}")
                return {
                    'error': 'AI返回格式异常',
                    'sequences': [],
                    'raw_response': content[:500]
                }
                
        except Exception as e:
            logger.error(f"AI序列设计失败: {e}")
            return {'error': str(e), 'sequences': []}
    
    def design_crispr_sequences(self, gene_name: str, gene_id: str = "", gene_description: str = "") -> Dict:
        """
        使用AI设计CRISPR/sgRNA序列，并返回推荐序列及来源文献/专利
        
        返回: {
            'sgrnas': [
                {
                    'sequence': '20nt靶序列（不含NGG）',
                    'pam': 'NGG',
                    'target_exon': '靶向外显子',
                    'design_tool': '设计工具（如Benchling、CRISPOR等）',
                    'efficiency_score': '效率评分',
                    'off_target_risk': '脱靶风险',
                    'references': [...]
                }
            ],
            'delivery_notes': '慢病毒载体递送建议',
            'validation_method': '验证方法'
        }
        """
        if not self.api_key:
            return {'error': '未配置AI API', 'sgrnas': []}
        
        try:
            prompt = f"""作为CRISPR基因编辑专家，请为基因"{gene_name}"（Gene ID: {gene_id}, 描述: {gene_description}）设计sgRNA序列。

请基于最新的文献和公开数据库知识，提供：
1. 3-4条高活性的sgRNA序列（20nt，不含PAM）
2. 每条序列的详细信息（靶向外显子、PAM序列、预测效率、脱靶风险）
3. 支持这些设计的参考文献或专利（如Brunello、GeCKO文库相关文献）

请按以下JSON格式回答（只返回JSON）：
{{
    "sgrnas": [
        {{
            "sequence": "GAGUCCGAGCAGAAGAAGAA",
            "pam": "NGG",
            "target_exon": "Exon 3",
            "cut_site": "距离起始密码子+156bp",
            "design_tool": "CRISPOR或Benchling算法",
            "efficiency_score": "高（预测>70% indel率）",
            "off_target_risk": "低（0-1个潜在脱靶位点）",
            "design_rationale": "靶向功能结构域，避免选择性剪接位点",
            "references": [
                {{
                    "type": "文献",
                    "title": "Optimized sgRNA design to maximize activity and minimize off-target effects of CRISPR-Cas9",
                    "authors": "Doench et al.",
                    "year": "2016",
                    "source": "Nature Biotechnology",
                    "pmid_or_patent": "PMID:26780180",
                    "url": "https://pubmed.ncbi.nlm.nih.gov/26780180/"
                }}
            ]
        }}
    ],
    "lentivirus_vector": {{
        "backbone": "lentiCRISPRv2或pLentiGuide-Puro",
        "promoter": "U6",
        "selection": "Puromycin或GFP",
        "cloning_strategy": "BsmBI酶切，粘性末端连接"
    }},
    "notes": "建议使用T7E1或Sanger测序验证切割效率；推荐设计非靶向对照sgRNA（NTC）",
    "validation_method": "T7E1酶切法或NGS测序检测indels，推荐在转导后72-96小时收集细胞检测"
}}

要求：
1. 序列必须符合SpCas9的NGG PAM要求
2. 优先选择靶向编码区外显子（特别是早期外显子或功能结构域）的序列
3. 参考文献必须是真实存在的（如Doench 2016、Sanjana 2014等CRISPR经典文献）
4. 提供具体的慢病毒载体构建建议"""

            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                'model': 'qwen-turbo',
                'input': {
                    'messages': [
                        {'role': 'system', 'content': '你是CRISPR-Cas9基因编辑专家，精通sgRNA序列设计和慢病毒递送系统，熟悉相关领域的经典文献和专利。'},
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
            content = result.get('output', {}).get('choices', [{}])[0].get('message', '').get('content', '')
            
            try:
                content_clean = content.replace('```json', '').replace('```', '').strip()
                design_data = json.loads(content_clean)
                return design_data
            except json.JSONDecodeError:
                logger.error(f"AI CRISPR设计返回非JSON格式: {content}")
                return {
                    'error': 'AI返回格式异常',
                    'sgrnas': [],
                    'raw_response': content[:500]
                }
                
        except Exception as e:
            logger.error(f"AI CRISPR设计失败: {e}")
            return {'error': str(e), 'sgrnas': []}

# ==================== 核心数据库（第一层） ====================
class CoreDatabases:
    """
    核心数据库 - 仅包含高置信度、跨细胞系保守的基因
    标准：DepMap核心必需基因 + 经典机制明确的毒性/抗病毒基因
    """
    
    # DepMap核心必需基因（基于CRISPR筛选数千细胞系，置信度>95%）
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
    
    # 核心毒性基因（机制明确，过表达直接导致细胞死亡）
    CORE_TOXIC = {
        'BAX': ('PMID:10625696', '促凋亡Bcl-2家族', '过表达直接激活线粒体凋亡途径'),
        'BAK1': ('PMID:10625696', '促凋亡Bcl-2家族', '线粒体外膜通透化诱导凋亡'),
        'BID': ('PMID:10625696', '促凋亡BH3-only蛋白', '连接死亡受体与线粒体凋亡'),
        'PUMA': ('PMID:12968034', 'p53下游促凋亡', '强力促凋亡BH3-only蛋白'),
        'NOXA': ('PMID:12968034', 'p53下游促凋亡', '促凋亡BH3-only蛋白'),
        'CASP3': ('PMID:9228057', '凋亡执行caspase', '过表达直接激活凋亡级联反应'),
        'CASP7': ('PMID:9228057', '凋亡执行caspase', '细胞凋亡执行分子'),
        'CASP8': ('PMID:9228057', '凋亡启动caspase', '死亡受体通路启动分子'),
        'CASP9': ('PMID:9228057', '凋亡启动caspase', '线粒体通路启动分子'),
        'FAS': ('PMID:8666142', '死亡受体', '激活外源性凋亡途径'),
        'TNF': ('PMID:15157675', '促炎细胞因子', '诱导细胞坏死性凋亡'),
        'TRAIL': ('PMID:10578115', 'TNF家族凋亡诱导配体', '选择性诱导肿瘤细胞凋亡'),
        'TP53': ('PMID:20154749', '肿瘤抑制因子', '过表达诱导G1阻滞和凋亡'),
        'CDKN1A': ('PMID:8242752', '细胞周期抑制因子', 'p21强抑制剂导致细胞周期停滞'),
        'PARP1': ('PMID:16794554', 'DNA修复酶', '过度激活导致NAD+耗竭和细胞死亡'),
    }
    
    # 核心抗病毒基因（干扰素刺激基因，明确抑制病毒复制）
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
        'TRIM5': ('PMID:15890885', '限制因子', '限制HIV-1等逆转录病毒复制'),
        'APOBEC3G': ('PMID:12134021', '限制因子', '胞嘧啶脱氨酶抑制HIV-1（Vif靶向）'),
        'BST2': ('PMID:19543227', '限制因子', 'Tetherin限制病毒出芽'),
        'KLF5': ('PMID:33597534', '转录因子-间接抗病毒', '调控IFITM家族和鞘脂代谢抑制SARS-CoV-2等病毒复制'),
    }
    
    @classmethod
    def check_gene(cls, gene_name: str, check_type: str) -> Optional[Tuple[str, str, str]]:
        """查询基因是否在核心列表中"""
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

# ==================== 密码验证（Streamlit Secrets）====================
class AuthManager:
    """密码验证 - 从Streamlit Secrets获取密码"""
    
    @staticmethod
    def check_password():
        """检查密码 - 使用Streamlit Secrets中的密码"""
        def password_entered():
            # 从secrets获取密码，如果没有设置则使用默认密码
            correct_password = st.secrets.get("APP_PASSWORD", "default123")
            if st.session_state["password"] == correct_password:
                st.session_state["password_correct"] = True
                del st.session_state["password"]
            else:
                st.session_state["password_correct"] = False

        if "password_correct" not in st.session_state:
            st.text_input("请输入访问密码", type="password", on_change=password_entered, key="password")
            st.info("🔒 请输入密码以访问系统")
            return False
        elif not st.session_state["password_correct"]:
            st.text_input("请输入访问密码", type="password", on_change=password_entered, key="password")
            st.error("😕 密码错误，请重试")
            return False
        else:
            return True

# ==================== HPA数据管理 ====================
class HPADataManager:
    """Human Protein Atlas数据管理 - 本地文件+SQLite缓存"""
    HPA_URL = "https://www.proteinatlas.org/download/proteinatlas.tsv.zip"
    LOCAL_DIR = "hpa_data"
    DB_FILE = "hpa_cache.db"
    
    def __init__(self):
        self.local_dir = self.LOCAL_DIR
        self.db_path = os.path.join(self.local_dir, self.DB_FILE)
        self.data_file = os.path.join(self.local_dir, "proteinatlas.tsv")
        self._init_storage()
    
    def _init_storage(self):
        if not os.path.exists(self.local_dir):
            os.makedirs(self.local_dir)
        if not os.path.exists(self.db_path):
            self._init_database()
    
    def _init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cell_line_expression (
                gene_name TEXT,
                cell_line TEXT,
                rna_level TEXT,
                protein_level TEXT,
                reliability TEXT,
                last_updated TIMESTAMP,
                PRIMARY KEY (gene_name, cell_line)
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()
    
    def check_and_download(self):
        """检查并下载HPA数据"""
        needs_download = False
        if not os.path.exists(self.data_file):
            needs_download = True
        else:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT value, updated_at FROM metadata WHERE key='last_check'")
            result = cursor.fetchone()
            conn.close()
            if result:
                last_check = datetime.fromisoformat(result[1])
                if datetime.now() - last_check > timedelta(days=30):
                    needs_download = True
            else:
                needs_download = True
        
        if needs_download:
            self._download_hpa_data()
    
    def _download_hpa_data(self):
        """下载HPA数据库"""
        try:
            st.info("📥 首次运行：正在下载HPA数据库（约200MB），请稍候...")
            zip_path = os.path.join(self.local_dir, "proteinatlas.tsv.zip")
            response = requests.get(self.HPA_URL, stream=True, timeout=300)
            response.raise_for_status()
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.local_dir)
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO metadata (key, value, updated_at) VALUES (?, ?, ?)",
                ('last_check', datetime.now().isoformat(), datetime.now().isoformat())
            )
            conn.commit()
            conn.close()
            os.remove(zip_path)
            st.success("✅ HPA数据下载完成")
        except Exception as e:
            logger.error(f"HPA download error: {e}")
            st.error(f"⚠️ HPA数据下载失败: {str(e)}")
    
    def get_expression_data(self, gene_name: str, cell_line: str) -> Optional[Dict]:
        """获取表达数据"""
        # 先检查缓存
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT rna_level, protein_level, reliability FROM cell_line_expression WHERE gene_name=? AND cell_line=?",
            (gene_name.upper(), cell_line.upper())
        )
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'rna_level': result[0],
                'protein_level': result[1],
                'reliability': result[2],
                'source': 'cache'
            }
        
        # 从本地文件查询
        if os.path.exists(self.data_file):
            return self._query_local_file(gene_name, cell_line)
        return None
    
    def _query_local_file(self, gene_name: str, cell_line: str) -> Optional[Dict]:
        """从本地TSV文件查询"""
        try:
            import csv
            gene_upper = gene_name.upper()
            cell_upper = cell_line.upper()
            
            with open(self.data_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter='\t')
                headers = reader.fieldnames
                
                # 查找匹配的细胞系列
                rna_col = None
                protein_col = None
                for header in headers:
                    header_upper = header.upper()
                    if 'RNA' in header_upper and cell_upper.replace(' ', '') in header_upper.replace(' ', '').replace('_', ''):
                        rna_col = header
                    if 'PROTEIN' in header_upper and cell_upper.replace(' ', '') in header_upper.replace(' ', '').replace('_', ''):
                        protein_col = header
                
                for row in reader:
                    if row.get('Gene', '').upper() == gene_upper or row.get('Gene name', '').upper() == gene_upper:
                        rna_level = row.get(rna_col, 'Not detected') if rna_col else 'Not detected'
                        prot_level = row.get(protein_col, 'Not detected') if protein_col else 'Not detected'
                        result = {
                            'rna_level': rna_level,
                            'protein_level': prot_level,
                            'reliability': 'Supported',
                            'source': 'hpa_file'
                        }
                        self._cache_result(gene_name, cell_line, result)
                        return result
            return None
        except Exception as e:
            logger.error(f"HPA query error: {e}")
            return None
    
    def _cache_result(self, gene_name: str, cell_line: str, data: Dict):
        """缓存查询结果"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO cell_line_expression 
                (gene_name, cell_line, rna_level, protein_level, reliability, last_updated)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                gene_name.upper(),
                cell_line.upper(),
                data.get('rna_level', 'Not detected'),
                data.get('protein_level', 'Not detected'),
                data.get('reliability', 'Unknown'),
                datetime.now().isoformat()
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Cache write error: {e}")

# ==================== API配置 ====================
class APIConfig:
    """API配置管理 - 从Streamlit Secrets获取"""
    
    @staticmethod
    def get_ncbi_credentials():
        """获取NCBI凭证 - 优先使用用户输入，其次使用Secrets"""
        user_email = st.session_state.get('ncbi_email_input', '').strip()
        user_key = st.session_state.get('ncbi_key_input', '').strip()
        secret_email = st.secrets.get("NCBI_EMAIL", "")
        secret_key = st.secrets.get("NCBI_API_KEY", "")
        
        email = user_email if user_email else secret_email
        api_key = user_key if user_key else secret_key
        
        if not email or email == "user@example.com":
            if not secret_email or secret_email == "user@example.com":
                return None, None, "API失效，需要填入有效API"
        
        return email, api_key, None
    
    @staticmethod
    def get_qwen_api_key():
        """获取通义千问API Key"""
        user_key = st.session_state.get('qwen_key_input', '').strip()
        secret_key = st.secrets.get("DASHSCOPE_API_KEY") or st.secrets.get("QWEN_API_KEY")
        return user_key if user_key else secret_key

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
    
    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_gene_data(_self, gene_name: str, organism: str) -> Tuple[Dict, List[Dict]]:
        """获取基因数据"""
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
    
    def _fetch_transcripts(self, gene_id: str) -> List[Dict]:
        """获取转录本数据"""
        try:
            search_params = {
                'db': 'nuccore',
                'term': f"{gene_id}[GeneID] AND (NM_[Title] OR XM_[Title])",
                'retmode': 'json',
                'retmax': 10
            }
            result = self._make_request('esearch.fcgi', search_params)
            if not result:
                return []
            
            ids = result.get('esearchresult', {}).get('idlist', [])
            if not ids:
                return []
            
            summary_params = {'db': 'nuccore', 'id': ','.join(ids), 'retmode': 'json'}
            result = self._make_request('esummary.fcgi', summary_params)
            if not result:
                return []
            
            docs = result.get('result', {})
            transcripts = []
            for uid in ids:
                try:
                    doc = docs.get(uid, {})
                    acc = doc.get('accessionversion', '')
                    length = doc.get('slen', 0)
                    if (acc.startswith('NM_') or acc.startswith('XM_')) and length > 0:
                        transcripts.append({
                            'id': acc,
                            'length': int(length),
                            'title': str(doc.get('title', ''))[:200]
                        })
                except Exception:
                    continue
            return transcripts
        except Exception as e:
            logger.error(f"Transcript fetch error: {e}")
            return []
    
    def search_gene_function_literature(self, gene_name: str, query_type: str) -> List[Dict]:
        """检索基因功能的特定类型文献"""
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
        """检索基因特定属性的文献 - 增强版（修复抗病毒检索）"""
        # 扩展检索策略：多层级覆盖直接和间接机制
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
            # 抗病毒检索策略：4层级全面覆盖
            'antiviral': [
                # 层级1：经典ISG/干扰素通路（原有）
                f"{gene_name} interferon antiviral innate immunity",
                f"{gene_name} virus replication restriction factor", 
                f"{gene_name} ISG interferon stimulated",
                
                # 层级2：间接抗病毒机制（转录因子、宿主因子）
                f"{gene_name} IFITM antiviral",  # KLF5调控IFITM家族
                f"{gene_name} transcription factor antiviral gene",
                f"{gene_name} regulates interferon stimulated gene",
                f"{gene_name} host factor virus infection",
                
                # 层级3：具体通路和病毒（扩大覆盖）
                f"{gene_name} STING MDA5 RIG-I pathway",
                f"{gene_name} influenza HIV SARS-CoV-2",
                f"{gene_name} virus susceptibility resistance",
                f"{gene_name} viral infection immune response",
                
                # 层级4：广义抗病毒相关（兜底）
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
                        abstract = doc.get('abstract', '') or doc.get('sorttitle', '')
                        if not title:
                            continue
                        
                        all_papers.append({
                            'pmid': str(pmid),
                            'title': html.escape(str(title)[:300]),
                            'abstract': html.escape(str(abstract)[:800]) if abstract else "[无摘要]",
                            'query': query,  # 记录来源查询词
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
        """检索细胞慢病毒感染参数"""
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
        """检索细胞转染条件"""
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
        """检索同细胞同基因研究"""
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
    """
    混合架构：
    第一层：核心数据库（DepMap + 经典文献，高置信度，瞬时响应）
    第二层：文献补充（PubMed检索 + AI语义分析，发现新证据，动态更新）
    """
    
    def __init__(self, ncbi_client: NCBIClient, ai_client: Optional[AIAnalysisClient] = None):
        self.ncbi = ncbi_client
        self.ai = ai_client
    
    def check_all(self, gene_name: str, transcripts: List[Dict], 
                  experiment_type: str) -> Tuple[bool, List[HardRuleCheck], Dict]:
        """执行混合检查"""
        checks = []
        evidence_summary = {
            'essential_checked': False,
            'toxic_checked': False,
            'antiviral_checked': False,
            'core_hits': [],
            'literature_hits': [],
            'ai_analyzed': False
        }
        
        # 1. 载体容量检查（仅过表达）
        if experiment_type.lower() == 'overexpression':
            check = self._check_vector_capacity(gene_name, transcripts)
            checks.append(check)
        
        # 2. 必需基因检查（仅敲除）
        if experiment_type.lower() == 'knockout':
            core_result = CoreDatabases.check_gene(gene_name, 'essential')
            if core_result:
                pmid, source, desc = core_result
                check = HardRuleCheck(
                    rule_name="必需基因检查（核心数据库）",
                    passed=False,
                    reason=f"❌ {gene_name}是{source}（{desc}），敲除可能导致细胞死亡",
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
        
        # 3. 毒性/抗病毒检查（仅过表达）
        if experiment_type.lower() == 'overexpression':
            # 检查核心毒性
            core_toxic = CoreDatabases.check_gene(gene_name, 'toxic')
            if core_toxic:
                pmid, source, desc = core_toxic
                check = HardRuleCheck(
                    rule_name="毒性基因检查（核心数据库）",
                    passed=False,
                    reason=f"❌ {gene_name}是{source}（{desc}），过表达可能导致细胞死亡",
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
            
            # 检查核心抗病毒
            core_antiviral = CoreDatabases.check_gene(gene_name, 'antiviral')
            if core_antiviral:
                # 修复：使用 core_antiviral 而不是 core_result
                pmid, source, desc = core_antiviral
                check = HardRuleCheck(
                    rule_name="抗病毒基因检查（核心数据库）",
                    passed=False,
                    reason=f"❌ {gene_name}是{source}（{desc}），过表达可能抑制慢病毒包装",
                    source=f"ISG数据库（{source}）",
                    pmid=pmid,
                    overrideable=False,
                    check_level="core"
                )
                checks.append(check)
                evidence_summary['antiviral_checked'] = True
                evidence_summary['core_hits'].append('antiviral')
            else:
                # 使用增强版文献检查（含AI分析）
                lit_check = self._check_by_literature_enhanced(gene_name, 'antiviral')
                checks.append(lit_check)
                if not lit_check.passed:
                    evidence_summary['antiviral_checked'] = True
                    evidence_summary['literature_hits'].append('antiviral')
                if self.ai and self.ai.api_key:
                    evidence_summary['ai_analyzed'] = True
        
        return all(c.passed for c in checks), checks, evidence_summary
    
    def _check_vector_capacity(self, gene_name: str, transcripts: List[Dict]) -> HardRuleCheck:
        """载体容量检查"""
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
                reason=f"✅ 转录本长度 {max_length}bp ≤2000bp，适合标准过表达",
                source="NCBI nuccore数据库",
                overrideable=True,
                check_level="core"
            )
        else:
            return HardRuleCheck(
                rule_name="载体容量检查（过表达）",
                passed=True,
                reason=f"⚠️ 转录本长度 {max_length}bp 超过2000bp，属于长序列过表达",
                source="NCBI nuccore数据库",
                pmid="PMID:15819909",
                overrideable=True,
                check_level="core"
            )
    
    def _check_by_literature(self, gene_name: str, check_type: str) -> HardRuleCheck:
        """文献补充检查 - 基础版（非抗病毒检查使用）"""
        papers = self.ncbi.search_gene_property_literature(gene_name, check_type)
        
        if not papers:
            type_names = {'essential': '必需性', 'toxic': '毒性', 'antiviral': '抗病毒功能'}
            return HardRuleCheck(
                rule_name=f"{type_names[check_type]}检查（文献补充）",
                passed=True,
                reason=f"✅ 核心数据库未收录，且未检索到相关文献",
                source="核心数据库+PubMed（零结果）",
                overrideable=True,
                check_level="literature"
            )
        
        # 基于文献原文的智能匹配（区分不同检查类型）
        type_names = {'essential': '必需性', 'toxic': '毒性', 'antiviral': '抗病毒功能'}
        
        if check_type == 'antiviral':
            # 抗病毒检查：扩展关键词库 + 模糊匹配 + 评分机制
            antiviral_keywords = {
                'high_confidence': [
                    'inhibit virus', 'inhibit viral', 'antiviral', 'anti-viral', 
                    'restrict virus', 'viral restriction', 'virus restriction', 
                    'resist virus', 'viral resistance', 'host restriction'
                ],
                'immune_pathway': [
                    'innate immunity', 'interferon', 'ifn', 'isg', 'type i interferon',
                    'immune response', 'immune defense', 'pathogen associated', 
                    'pattern recognition', 'prr', 'tlr', 'rig-i', 'mda5', 'sting',
                    'ifitm', 'isg15', 'isg56', ' innate immune '
                ],
                'virus_specific': [
                    'influenza', 'hiv', 'sars-cov', 'coronavirus', 'herpes', 
                    'hepatitis', 'vesicular stomatitis', 'vsv', ' EMCV '
                ],
                'mechanism': [
                    'host factor', 'susceptibility', 'resistance to infection', 
                    'viral replication', 'virus entry', 'virus assembly', 
                    'virus budding', 'interferon stimulated'
                ],
                'transcription_regulation': [
                    'transcription factor', 'regulates expression', 'promoter activity',
                    'gene regulation', 'upregulates', 'downregulates', 'induces expression'
                ]
            }
            
            # 扁平化所有关键词用于匹配
            all_keywords = []
            for category, words in antiviral_keywords.items():
                all_keywords.extend(words)
            
            evidence = []
            for paper in papers:
                text = (paper.get('abstract', '') + ' ' + paper.get('title', '')).lower()
                
                # 计算匹配得分
                match_score = 0
                matched_terms = []
                
                for term in all_keywords:
                    if term in text:
                        match_score += 1
                        matched_terms.append(term)
                
                # 特殊加分：标题中出现关键词
                title_lower = paper.get('title', '').lower()
                if any(term in title_lower for term in ['antiviral', 'virus', 'interferon', 'innate immunity']):
                    match_score += 2
                
                # 判定标准：匹配分>=3 或 明确包含病毒名称+免疫相关词
                if match_score >= 3:
                    evidence.append({
                        **paper, 
                        'match_score': match_score,
                        'matched_terms': matched_terms[:5]  # 记录前5个匹配词
                    })
                elif match_score >= 1 and any(v in text for v in ['virus', 'viral', 'infection']):
                    # 次要标准：只要有病毒相关词且至少有1个关键词匹配
                    evidence.append({
                        **paper, 
                        'match_score': match_score,
                        'matched_terms': matched_terms[:5]
                    })
            
            # 按匹配度排序
            evidence.sort(key=lambda x: x.get('match_score', 0), reverse=True)
            
        else:
            # 必需性和毒性检查：保留原有的严格匹配逻辑
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
            
            # 构建详细的匹配说明
            if check_type == 'antiviral' and 'match_score' in best_match:
                match_info = f"（匹配度: {best_match['match_score']}/12，关键词: {', '.join(best_match.get('matched_terms', [])[:3])}）"
            else:
                match_info = ""
            
            return HardRuleCheck(
                rule_name=f"{type_names[check_type]}检查（文献补充）",
                passed=False,
                reason=f"❌ 文献检索发现 {gene_name} 具有{type_names[check_type]}证据：{best_match['title'][:80]}...{match_info}",
                source=f"PubMed文献检索（{len(evidence)}篇明确证据，核心数据库未收录）",
                pmid=pmid_list[0],
                pmid_list=pmid_list,
                evidence_papers=evidence[:3],
                overrideable=False,
                check_level="literature"
            )
        
        # 无明确证据但检索到文献的情况
        return HardRuleCheck(
            rule_name=f"{type_names[check_type]}检查（文献补充）",
            passed=True,
            reason=f"✅ 检索到{len(papers)}篇文献，但未发现明确{type_names[check_type]}证据",
            source="PubMed文献检索（无明确证据）",
            pmid_list=[p['pmid'] for p in papers[:3]],
            evidence_papers=papers[:3],
            overrideable=True,
            check_level="literature"
        )
    
    def _check_by_literature_enhanced(self, gene_name: str, check_type: str) -> HardRuleCheck:
        """文献补充检查 - 增强版（含AI语义分析，仅用于抗病毒检查）"""
        if check_type != 'antiviral':
            return self._check_by_literature(gene_name, check_type)
        
        papers = self.ncbi.search_gene_property_literature(gene_name, check_type)
        
        if not papers:
            return HardRuleCheck(
                rule_name="抗病毒基因检查（文献+AI分析）",
                passed=True,
                reason=f"✅ 核心数据库未收录，且未检索到相关文献",
                source="核心数据库+PubMed（零结果）",
                overrideable=True,
                check_level="literature"
            )
        
        # 第一步：传统关键词筛选（减少AI调用量）
        pre_filtered = []
        for paper in papers[:10]:  # 只分析前10篇最相关的
            text = (paper.get('abstract', '') + ' ' + paper.get('title', '')).lower()
            # 宽松预筛：只要提到病毒相关就保留
            if any(kw in text for kw in ['virus', 'viral', 'infection', 'interferon', 'immunity', 'host']):
                pre_filtered.append(paper)
        
        if not pre_filtered:
            return HardRuleCheck(
                rule_name="抗病毒基因检查（文献+AI分析）",
                passed=True,
                reason=f"✅ 检索到{len(papers)}篇文献，但预筛选未发现有潜力文献",
                source="PubMed文献检索（关键词预筛选）",
                pmid_list=[p['pmid'] for p in papers[:3]],
                evidence_papers=papers[:3],
                overrideable=True,
                check_level="literature"
            )
        
        # 第二步：AI语义分析（如果配置了API）
        ai_evidence = []
        if self.ai and self.ai.api_key:
            with st.spinner(f"🤖 AI正在分析{len(pre_filtered)}篇文献的抗病毒证据..."):
                for paper in pre_filtered[:3]:  # AI分析前3篇
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
        
        # 第三步：综合判定
        if ai_evidence:
            best = ai_evidence[0]
            pmid_list = [p['pmid'] for p in ai_evidence[:3]]
            
            mechanism = best.get('ai_mechanism', '未知机制')
            confidence = best.get('ai_confidence', 0)
            
            return HardRuleCheck(
                rule_name="抗病毒基因检查（文献+AI分析）",
                passed=False,
                reason=f"❌ AI分析确认 {gene_name} 具有抗病毒功能（置信度: {confidence:.0%}）",
                source=f"PubMed + 通义千问AI语义分析（机制: {mechanism}）",
                pmid=best['pmid'],
                pmid_list=pmid_list,
                evidence_papers=ai_evidence[:3],
                overrideable=False,
                check_level="literature"
            )
        
        # AI未确认，回退到关键词匹配
        return self._check_by_literature(gene_name, check_type)

# ==================== 基因输入组件 ====================
class GeneAutocompleteService:
    def __init__(self):
        self.clinical_tables_url = "https://clinicaltables.nlm.nih.gov/api/ncbi_genes/v3/search"
    
    @st.cache_data(ttl=3600, show_spinner=False)
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
    def __init__(self, gene_service: GeneAutocompleteService):
        self.gene_service = gene_service
    
    def render(self, organism: str, key_prefix: str = "gene") -> Optional[str]:
        input_key = f"{key_prefix}_input"
        selected_key = f"{key_prefix}_selected"
        suggestions_key = f"{key_prefix}_suggestions"
        last_query_key = f"{key_prefix}_last_query"
        
        if input_key not in st.session_state:
            st.session_state[input_key] = ""
        if selected_key not in st.session_state:
            st.session_state[selected_key] = ""
        if suggestions_key not in st.session_state:
            st.session_state[suggestions_key] = []
        if last_query_key not in st.session_state:
            st.session_state[last_query_key] = ""
        
        user_input = st.text_input(
            "基因名（支持自动完成，输入2个字符以上显示建议）",
            value=st.session_state[input_key],
            key=f"{key_prefix}_text_widget"
        )
        
        if user_input != st.session_state[input_key]:
            st.session_state[input_key] = user_input
            if st.session_state[selected_key] and user_input != st.session_state[selected_key]:
                st.session_state[selected_key] = ""
                st.session_state[suggestions_key] = []
            if len(user_input) >= 2:
                st.rerun()
        
        if len(user_input) >= 2 and not st.session_state[selected_key]:
            last_query = st.session_state.get(last_query_key, "")
            if user_input != last_query:
                suggestions = self.gene_service.get_suggestions(user_input, organism)
                st.session_state[suggestions_key] = suggestions
                st.session_state[last_query_key] = user_input
                st.rerun()
        
        suggestions = st.session_state.get(suggestions_key, [])
        if suggestions and not st.session_state[selected_key]:
            cols = st.columns(min(len(suggestions), 4))
            for i, gene in enumerate(suggestions):
                with cols[i % 4]:
                    display_text = f"{gene['symbol']}"
                    if st.button(display_text, key=f"{key_prefix}_sug_{i}", use_container_width=True):
                        st.session_state[selected_key] = gene['symbol']
                        st.session_state[input_key] = gene['symbol']
                        st.session_state[f"{key_prefix}_info"] = gene
                        st.rerun()
        
        if st.session_state[selected_key] and f"{key_prefix}_info" in st.session_state:
            gene_info = st.session_state[f"{key_prefix}_info"]
            st.success(f"✅ 已选择: **{gene_info['symbol']}** | {gene_info.get('name', '')} | {gene_info.get('chromosome', '')}")
        
        if st.session_state[selected_key]:
            return st.session_state[selected_key]
        elif user_input:
            return user_input
        return None

# ==================== 报告导出 ====================
class ReportExporter:
    @staticmethod
    def generate_html_report(result: Dict) -> str:
        """生成HTML报告"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>慢病毒包装与细胞系构建评估报告</title>
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
                <h1>慢病毒包装与细胞系构建评估报告</h1>
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
        """生成CSV报告"""
        output = StringIO()
        writer = pd.DataFrame([{
            '基因': result.get('gene', ''),
            '物种': result.get('organism', ''),
            '细胞系': result.get('cell_line', ''),
            '实验类型': result.get('experiment', ''),
            '评估结论': result.get('final_recommendation', ''),
            '评估依据': result.get('primary_basis', ''),
            '生成时间': result.get('timestamp', '')
        }])
        return writer.to_csv(index=False)

# ==================== 主评估引擎 ====================
class HybridAssessmentEngine:
    def __init__(self, email: str, ncbi_api_key: Optional[str] = None, ai_api_key: Optional[str] = None):
        self.ncbi = NCBIClient(email, ncbi_api_key)
        self.ai = AIAnalysisClient(ai_api_key) if ai_api_key else None
        self.hard_rules = HybridHardRulesEngine(self.ncbi, self.ai)
        self.hpa = HPADataManager()
    
    def assess(self, gene_name: str, organism: str, cell_line: Optional[str], 
               experiment_type: str) -> Dict:
        result = {
            'timestamp': datetime.now().isoformat(),
            'gene': gene_name,
            'organism': organism,
            'cell_line': cell_line,
            'experiment': experiment_type,
            'decision_hierarchy': {},
            'final_recommendation': '',
            'primary_basis': ''
        }
        
        # 获取基因数据
        with st.spinner("🔍 检索基因详细数据..."):
            gene_info, transcripts = self.ncbi.fetch_gene_data(gene_name, organism)
        
        if not gene_info:
            return {'error': f'无法获取基因 {html.escape(gene_name)} 的信息'}
        
        result['gene_info'] = {
            'id': gene_info.get('id', ''),
            'name': gene_info.get('name', ''),
            'description': gene_info.get('description', '')[:200]
        }
        
        # 混合硬性规则检查（含AI分析）
        with st.spinner("⚙️ 执行混合硬性规则检查（含AI语义分析）..."):
            hard_passed, hard_checks, evidence_summary = self.hard_rules.check_all(
                gene_name, transcripts, experiment_type
            )
        
        result['decision_hierarchy']['hard_rules'] = {
            'passed': hard_passed,
            'checks': [asdict(c) for c in hard_checks],
            'evidence_summary': evidence_summary
        }
        
        # 检查阻断
        blocking = [c for c in hard_checks if not c.passed and not c.overrideable]
        if blocking:
            result['final_recommendation'] = 'BLOCKED'
            result['primary_basis'] = '硬性生物学限制（核心数据库或文献证据）'
            result['blocking_evidence'] = [asdict(c) for c in blocking]
            return result
        
        # 基因功能全面分析（新增模块）
        if self.ai and self.ai.api_key:
            with st.spinner("📚 AI正在分析基因功能及实验模型数据..."):
                # 检索不同类型的文献
                papers_general = self.ncbi.search_gene_function_literature(gene_name, 'general')
                papers_oe = self.ncbi.search_gene_function_literature(gene_name, 'overexpression')
                papers_kd = self.ncbi.search_gene_function_literature(gene_name, 'knockdown')
                papers_ko = self.ncbi.search_gene_function_literature(gene_name, 'knockout')
                
                # AI综合分析
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
                    'source': 'AI基于文献综合分析'
                }
        
        # HPA表达数据（仅Human + 输入细胞名）
        if organism == 'Homo sapiens' and cell_line:
            with st.spinner("🧬 查询HPA表达数据..."):
                hpa_data = self.hpa.get_expression_data(gene_name, cell_line)
                result['hpa_data'] = hpa_data or {
                    'rna_level': '数据难以获得',
                    'protein_level': '数据难以获得',
                    'reliability': 'N/A'
                }
        
        # 细胞评估数据
        if cell_line:
            with st.spinner("🧫 检索细胞相关参数..."):
                cell_params = self.ncbi.search_cell_lentivirus_params(cell_line)
                transfection_params = self.ncbi.search_cell_transfection(cell_line)
                same_cell_studies = self.ncbi.search_same_cell_gene_studies(gene_name, cell_line)
                result['cell_assessment'] = {
                    'lentivirus_params': cell_params if cell_params else '无已报道的参数',
                    'transfection_params': transfection_params if transfection_params else '无已报道的参数',
                    'same_cell_gene_studies': same_cell_studies if same_cell_studies else '无同细胞同基因研究报道'
                }
        
        # 序列设计检索（敲低/敲除）- 使用AI直接设计
        if experiment_type.lower() in ['knockdown', 'knockout']:
            if self.ai and self.ai.api_key:
                with st.spinner("🧬 AI正在设计序列并提供参考文献..."):
                    if experiment_type.lower() == 'knockdown':
                        # AI设计siRNA/shRNA
                        design_data = self.ai.design_rnai_sequences(
                            gene_name=gene_name,
                            gene_id=gene_info.get('id', ''),
                            gene_description=gene_info.get('description', '')
                        )
                        result['sequence_designs'] = {
                            'type': 'siRNA/shRNA (AI设计)',
                            'designs': design_data,
                            'source': 'AI基于最新文献和数据库知识设计'
                        }
                    else:
                        # AI设计sgRNA
                        design_data = self.ai.design_crispr_sequences(
                            gene_name=gene_name,
                            gene_id=gene_info.get('id', ''),
                            gene_description=gene_info.get('description', '')
                        )
                        result['sequence_designs'] = {
                            'type': 'sgRNA (AI设计)',
                            'designs': design_data,
                            'source': 'AI基于最新文献和数据库知识设计'
                        }
            else:
                # 未配置AI时回退到文献检索
                with st.spinner("🎯 检索序列设计文献（未配置AI，使用文献检索）..."):
                    if experiment_type.lower() == 'knockdown':
                        sequences = self.ncbi.search_sirna_sequences(gene_name)
                        result['sequence_designs'] = {
                            'type': 'siRNA/shRNA',
                            'designs': sequences if sequences else '无已报道的设计',
                            'source': 'PubMed文献检索'
                        }
                    else:
                        sequences = self.ncbi.search_sgrna_sequences(gene_name)
                        result['sequence_designs'] = {
                            'type': 'sgRNA',
                            'designs': sequences if sequences else '无已报道的设计',
                            'source': 'PubMed文献检索'
                        }
        
        # 生成建议
        warning_checks = [c for c in hard_checks if not c.passed and c.overrideable]
        if warning_checks:
            result['final_recommendation'] = "⚠️ 警告：检测到潜在风险，建议谨慎操作"
            result['primary_basis'] = f"基于{len(warning_checks)}项警告（可人工覆盖）"
        else:
            result['final_recommendation'] = "✅ 未检测到明确风险，可进行标准流程"
            result['primary_basis'] = "基于核心数据库筛查和文献检索"
        
        return result

# ==================== UI渲染 ====================
def render_sidebar():
    """渲染侧边栏"""
    with st.sidebar:
        st.header("⚙️ API配置")
        st.subheader("NCBI配置")
        ncbi_email = st.text_input("NCBI邮箱", value="", key="ncbi_email_input", 
                                   help="优先使用此处输入，留空使用Secrets")
        ncbi_key = st.text_input("NCBI API Key", type="password", key="ncbi_key_input", 
                                help="可选，用于提高访问频率限制")
        
        email, key, error = APIConfig.get_ncbi_credentials()
        if error:
            st.error(error)
        else:
            st.success("✅ NCBI API有效")
        
        st.divider()
        st.subheader("AI配置（可选）")
        qwen_key = st.text_input("通义千问API Key", type="password", key="qwen_key_input", 
                                help="可选，用于AI文献语义分析（增强抗病毒检测）和序列设计")
        
        final_qwen = APIConfig.get_qwen_api_key()
        if final_qwen:
            st.success("✅ AI API已配置 - 将启用语义分析和智能序列设计")
        else:
            st.info("ℹ️ 未配置AI（将使用关键词匹配和传统文献检索）")
        
        st.divider()
        st.caption("🔒 核心列表+文献补充+AI语义分析混合策略")

def render_main_panel():
    """渲染主面板"""
    st.markdown("""
    <h1 style='text-align: center; color: #1f77b4; margin-bottom: 30px;'>
        🧬 慢病毒包装与细胞系构建评估系统
    </h1>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🔬 实验参数输入")
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
        gene_service = GeneAutocompleteService()
        gene_component = GeneInputComponent(gene_service)
        gene = gene_component.render(organism, key_prefix="main_gene")
    
    with col2:
        cell_line = st.text_input("细胞名（可选）", placeholder="例如：HEK293T, HeLa, A549", 
                                 help="输入细胞系名称以获取特定细胞系的感染参数和HPA表达数据")
        exp_type = st.selectbox(
            "评估选项",
            ["overexpression", "knockdown", "knockout"],
            format_func=lambda x: {
                "overexpression": "过表达 (OE)",
                "knockdown": "敲低 (RNAi)",
                "knockout": "敲除 (CRISPR)"
            }.get(x, x)
        )
    
    analyze = st.button("🚀 开始AI智能评估", type="primary", use_container_width=True)
    return organism, gene, cell_line, exp_type, analyze

def render_results(result: Dict):
    """渲染评估结果"""
    if 'error' in result:
        st.error(result['error'])
        return
    
    st.divider()
    st.markdown(f"## 🎯 评估报告 - {html.escape(result['gene'])}")
    
    # 导出按钮
    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        exporter = ReportExporter()
        html_report = exporter.generate_html_report(result)
        st.download_button(
            "📄 导出HTML报告",
            html_report,
            file_name=f"assessment_{result['gene']}_{datetime.now().strftime('%Y%m%d')}.html",
            mime="text/html"
        )
    with col_exp2:
        csv_report = exporter.generate_csv_report(result)
        st.download_button(
            "📊 导出CSV报告",
            csv_report,
            file_name=f"assessment_{result['gene']}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    # 评估结论
    rec = result['final_recommendation']
    rec_color = {"❌": "#ffebee", "⚠️": "#fff3e0", "⚡": "#fff8e1", "✅": "#e8f5e9"}.get(rec[:2], "#f5f5f5")
    
    st.markdown(f"""
    <div style='padding: 20px; background-color: {rec_color}; 
                border-radius: 10px; text-align: center; margin: 20px 0;
                border: 2px solid #ddd;'>
        <h3>{html.escape(rec)}</h3>
        <small>{html.escape(result.get('primary_basis', ''))}</small>
    </div>
    """, unsafe_allow_html=True)
    
    # 标签页 - 新增"基因功能分析"
    tabs = st.tabs(["硬性规则检查", "基因功能分析", "HPA表达数据", "细胞评估", "序列设计"])
    
    with tabs[0]:
        st.markdown("### 🚦 混合硬性规则检查（核心数据库+文献补充+AI语义分析）")
        hierarchy = result.get('decision_hierarchy', {})
        hard_rules = hierarchy.get('hard_rules', {})
        evidence_summary = hard_rules.get('evidence_summary', {})
        
        # 显示检查统计
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("核心数据库命中", len(evidence_summary.get('core_hits', [])))
        with col2:
            st.metric("文献补充发现", len(evidence_summary.get('literature_hits', [])))
        with col3:
            st.metric("总检查项", len(hard_rules.get('checks', [])))
        with col4:
            ai_status = "✅" if evidence_summary.get('ai_analyzed') else "➖"
            st.metric("AI语义分析", ai_status)
        
        st.divider()
        
        for check in hard_rules.get('checks', []):
            icon = "✅" if check['passed'] else "❌" if not check['overrideable'] else "⚠️"
            color = "green" if check['passed'] else "red" if not check['overrideable'] else "orange"
            level_badge = "🔴 核心" if check.get('check_level') == "core" else "🔵 文献"
            
            evidence_md = ""
            if check.get('evidence_papers'):
                evidence_md = "<br/><small>📚 证据文献：</small><br/>"
                for paper in check['evidence_papers'][:2]:
                    pmid = paper.get('pmid', '')
                    title = paper.get('title', '')[:80]
                    
                    # 显示AI分析结果
                    ai_info = ""
                    if 'ai_confidence' in paper:
                        ai_info = f"<br/><small>🤖 AI置信度: {paper['ai_confidence']:.0%} | 机制: {paper.get('ai_mechanism', '未知')}</small>"
                    
                    match_info = ""
                    if 'match_score' in paper:
                        match_info = f" [匹配度:{paper['match_score']}]"
                    
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
    
    # 新增：基因功能分析标签页
    with tabs[1]:
        st.markdown("### 📖 基因功能全面分析（基于AI文献综合）")
        
        func_analysis = result.get('gene_function_analysis', {})
        if func_analysis and not func_analysis.get('data', {}).get('error'):
            data = func_analysis.get('data', {})
            lit_counts = func_analysis.get('literature_counts', {})
            
            # 文献统计
            st.markdown("#### 📊 文献覆盖统计")
            cols = st.columns(4)
            cols[0].metric("基因功能文献", lit_counts.get('general', 0))
            cols[1].metric("过表达研究", lit_counts.get('overexpression', 0))
            cols[2].metric("敲低研究", lit_counts.get('knockdown', 0))
            cols[3].metric("敲除研究", lit_counts.get('knockout', 0))
            st.caption(f"来源: {func_analysis.get('source', 'AI分析')}")
            st.divider()
            
            # 蛋白基础功能
            if 'protein_function' in data:
                with st.expander("🧬 蛋白基础功能", expanded=True):
                    pf = data['protein_function']
                    st.write(f"**蛋白类别**: {pf.get('category', 'N/A')}")
                    st.write(f"**结构域**: {pf.get('domains', 'N/A')}")
                    st.write(f"**信号通路**: {pf.get('pathways', 'N/A')}")
                    st.write(f"**亚细胞定位**: {pf.get('cellular_location', 'N/A')}")
                    st.write(f"**组织表达**: {pf.get('tissue_expression', 'N/A')}")
            
            # 过表达效应
            if 'overexpression' in data:
                with st.expander("⬆️ 过表达效应（Gain-of-function）"):
                    oe = data['overexpression']
                    if 'cell_models' in oe and oe['cell_models']:
                        st.markdown("**细胞模型:**")
                        for model in oe['cell_models']:
                            with st.container():
                                st.markdown(f"""
                                - **细胞系**: {model.get('cell_line', 'N/A')}  
                                  **表型**: {model.get('phenotype', 'N/A')}  
                                  **机制**: {model.get('mechanism', 'N/A')}  
                                  **文献**: {model.get('reference', 'N/A')}
                                """)
                    else:
                        st.info("未见细胞水平过表达报道")
                    
                    if 'animal_models' in oe and oe['animal_models']:
                        st.markdown("**动物模型:**")
                        for model in oe['animal_models']:
                            st.markdown(f"""
                            - **模型**: {model.get('model', 'N/A')}  
                              **表型**: {model.get('phenotype', 'N/A')}  
                              **文献**: {model.get('reference', 'N/A')}
                            """)
                    else:
                        st.info("未见动物模型过表达报道")
                    
                    if 'summary' in oe:
                        st.success(f"**总结**: {oe['summary']}")
            
            # 敲低效应
            if 'knockdown' in data:
                with st.expander("⬇️ 敲低效应（RNAi）"):
                    kd = data['knockdown']
                    if 'cell_models' in kd and kd['cell_models']:
                        st.markdown("**细胞模型:**")
                        for model in kd['cell_models']:
                            st.markdown(f"""
                            - **细胞系**: {model.get('cell_line', 'N/A')}  
                              **方法**: {model.get('method', 'N/A')}  
                              **表型**: {model.get('phenotype', 'N/A')}  
                              **文献**: {model.get('reference', 'N/A')}
                            """)
                    else:
                        st.info("未见敲低报道")
                    
                    if 'summary' in kd:
                        st.success(f"**总结**: {kd['summary']}")
            
            # 敲除效应
            if 'knockout' in data:
                with st.expander("🚫 敲除效应（Knockout）"):
                    ko = data['knockout']
                    
                    if 'cell_models' in ko and ko['cell_models']:
                        st.markdown("**细胞模型:**")
                        for model in ko['cell_models']:
                            st.markdown(f"""
                            - **细胞系**: {model.get('cell_line', 'N/A')}  
                              **方法**: {model.get('method', 'N/A')}  
                              **表型**: {model.get('phenotype', 'N/A')}  
                              **细胞活力影响**: {model.get('viability', 'N/A')}  
                              **文献**: {model.get('reference', 'N/A')}
                            """)
                    else:
                        st.info("未见细胞敲除报道")
                    
                    if 'animal_models' in ko and ko['animal_models']:
                        st.markdown("**动物模型:**")
                        for model in ko['animal_models']:
                            st.markdown(f"""
                            - **模型**: {model.get('model', 'N/A')}  
                              **表型**: {model.get('phenotype', 'N/A')}  
                              **致死性**: {model.get('lethality', 'N/A')}  
                              **文献**: {model.get('reference', 'N/A')}
                            """)
                    else:
                        st.info("未见动物敲除报道")
                    
                    if 'summary' in ko:
                        st.success(f"**总结**: {ko['summary']}")
            
            # 疾病相关性
            if 'disease_relevance' in data:
                with st.expander("🏥 疾病相关性"):
                    dr = data['disease_relevance']
                    st.write(f"**肿瘤作用**: {dr.get('cancer', 'N/A')}")
                    st.write(f"**其他疾病**: {dr.get('other_diseases', 'N/A')}")
                    st.write(f"**治疗潜力**: {dr.get('therapeutic_potential', 'N/A')}")
            
            # 实验注意事项
            if 'experimental_notes' in data:
                st.warning(f"⚠️ **实验设计注意事项**: {data['experimental_notes']}")
            
            # 关键文献
            if 'key_references' in data:
                with st.expander("📚 关键参考文献"):
                    for ref in data['key_references']:
                        st.markdown(f"- {ref}")
        
        else:
            st.info("⚠️ 未配置AI API或未获取到功能分析数据。请在侧边栏配置通义千问API Key以启用此功能。")
    
    with tabs[2]:
        st.markdown("### 🧬 HPA表达量数据")
        hpa_data = result.get('hpa_data')
        if hpa_data:
            if 'rna_level' in hpa_data:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("RNA水平", hpa_data.get('rna_level', 'N/A'))
                with col2:
                    st.metric("蛋白水平", hpa_data.get('protein_level', 'N/A'))
                with col3:
                    st.metric("可靠性", hpa_data.get('reliability', 'N/A'))
                gene = result['gene']
                st.markdown(f"[查看HPA详情](https://www.proteinatlas.org/{gene}-{result.get('cell_line', '')})")
            else:
                st.info(hpa_data.get('message', '数据难以获得'))
        else:
            st.info("⚠️ 仅当物种为人类且输入细胞名时显示HPA数据")
    
    with tabs[3]:
        st.markdown("### 🧫 细胞系构建评估数据")
        cell_data = result.get('cell_assessment')
        if cell_data:
            st.subheader("📚 同细胞同基因文献")
            studies = cell_data.get('same_cell_gene_studies')
            if studies and studies != '无同细胞同基因研究报道':
                for study in studies[:5]:
                    st.markdown(f"""
                    - **{html.escape(study.get('title', ''))}**  
                      {html.escape(study.get('journal', ''))} ({study.get('year', '')})  
                      [PMID: {study.get('pmid', '')}]({study.get('url', '')})
                    """)
            else:
                st.warning(studies if isinstance(studies, str) else "无已报道的研究")
            
            st.subheader("🦠 慢病毒MOI参数")
            params = cell_data.get('lentivirus_params')
            if params and params != '无已报道的参数':
                for param in params[:5]:
                    st.markdown(f"- [PMID: {param.get('pmid', '')}]({param.get('url', '')}) - {param.get('note', '')}")
            else:
                st.warning("无已报道的参数")
            
            st.subheader("⚡ 转染/电转条件")
            trans = cell_data.get('transfection_params')
            if trans and trans != '无已报道的参数':
                for t in trans[:5]:
                    st.markdown(f"- [PMID: {t.get('pmid', '')}]({t.get('url', '')}) - {t.get('note', '')}")
            else:
                st.warning("无已报道的参数")
        else:
            st.info("⚠️ 输入细胞名以获取细胞评估数据")
    
    with tabs[4]:
        st.markdown("### 🎯 序列设计参考")
        seq_data = result.get('sequence_designs')
        if seq_data:
            st.subheader(f"{seq_data.get('type', '')}")
            
            # 显示数据来源
            source_info = seq_data.get('source', '')
            if 'AI' in source_info:
                st.success(f"🤖 {source_info}")
            else:
                st.info(f"📚 {source_info}")
            
            designs = seq_data.get('designs', {})
            
            # 处理AI设计的序列
            if isinstance(designs, dict) and not designs.get('error'):
                if 'sequences' in designs:
                    # siRNA设计展示
                    for i, seq in enumerate(designs.get('sequences', [])):
                        with st.expander(f"候选序列 #{i+1}: {seq.get('target_seq', 'N/A')}"):
                            st.write(f"**靶序列**: `{seq.get('target_seq', 'N/A')}`")
                            st.write(f"**靶向区域**: {seq.get('target_region', 'N/A')}")
                            st.write(f"**设计原理**: {seq.get('design_rationale', 'N/A')}")
                            st.write(f"**预期效率**: {seq.get('efficiency_score', 'N/A')}")
                            
                            # 显示参考文献
                            refs = seq.get('references', [])
                            if refs:
                                st.write("**参考文献/专利**:")
                                for ref in refs:
                                    ref_type = ref.get('type', '文献')
                                    with st.container():
                                        col1, col2 = st.columns([3, 1])
                                        with col1:
                                            st.markdown(f"*{html.escape(ref.get('title', ''))}*  \n"
                                                      f"{html.escape(ref.get('authors', ''))} ({ref.get('year', '')})  \n"
                                                      f"*{html.escape(ref.get('source', ''))}*")
                                        with col2:
                                            pmid = ref.get('pmid_or_patent', '')
                                            url = ref.get('url', '')
                                            if url:
                                                st.markdown(f"[{pmid}]({url})")
                                            else:
                                                st.text(pmid)
                
                elif 'sgrnas' in designs:
                    # sgRNA设计展示
                    for i, sgrna in enumerate(designs.get('sgrnas', [])):
                        with st.expander(f"sgRNA #{i+1}: {sgrna.get('sequence', 'N/A')}"):
                            st.write(f"**靶序列**: `{sgrna.get('sequence', 'N/A')}`")
                            st.write(f"**PAM**: {sgrna.get('pam', 'NGG')}")
                            st.write(f"**靶向外显子**: {sgrna.get('target_exon', 'N/A')}")
                            st.write(f"**切割位点**: {sgrna.get('cut_site', 'N/A')}")
                            st.write(f"**预测效率**: {sgrna.get('efficiency_score', 'N/A')}")
                            st.write(f"**脱靶风险**: {sgrna.get('off_target_risk', 'N/A')}")
                            st.write(f"**设计依据**: {sgrna.get('design_rationale', 'N/A')}")
                            
                            # 显示参考文献
                            refs = sgrna.get('references', [])
                            if refs:
                                st.write("**参考文献/专利**:")
                                for ref in refs:
                                    ref_type = ref.get('type', '文献')
                                    with st.container():
                                        col1, col2 = st.columns([3, 1])
                                        with col1:
                                            st.markdown(f"*{html.escape(ref.get('title', ''))}*  \n"
                                                      f"{html.escape(ref.get('authors', ''))} ({ref.get('year', '')})  \n"
                                                      f"*{html.escape(ref.get('source', ''))}*")
                                        with col2:
                                            pmid = ref.get('pmid_or_patent', '')
                                            url = ref.get('url', '')
                                            if url:
                                                st.markdown(f"[{pmid}]({url})")
                                            else:
                                                st.text(pmid)
                
                # 显示载体设计建议
                if 'shrna_vector_design' in designs:
                    vec = designs['shrna_vector_design']
                    st.divider()
                    st.subheader("🧬 shRNA载体设计建议")
                    st.json(vec)
                
                if 'lentivirus_vector' in designs:
                    vec = designs['lentivirus_vector']
                    st.divider()
                    st.subheader("🧬 慢病毒载体建议")
                    st.json(vec)
                
                # 显示通用注意事项
                if 'notes' in designs:
                    st.info(f"💡 **注意事项**: {designs['notes']}")
                
                if 'validation_method' in designs:
                    st.success(f"✅ **验证建议**: {designs['validation_method']}")
            
            # 处理传统文献检索结果
            elif isinstance(designs, list):
                if designs and designs != '无已报道的设计':
                    for design in designs[:5]:
                        st.markdown(f"""
                        - **类型**: {design.get('type', '')}  
                          **说明**: {design.get('note', '')}  
                          [查看文献 (PMID: {design.get('pmid', '')})]({design.get('url', '')})
                        """)
                else:
                    st.warning("无已报道的设计")
            
            else:
                st.error("序列设计数据格式异常")
        else:
            st.info("⚠️ 敲低和敲除实验可查看序列设计建议")

def main():
    """主函数"""
    # 密码验证
    if not AuthManager.check_password():
        st.stop()
    
    # 初始化HPA数据管理器
    hpa_manager = HPADataManager()
    hpa_manager.check_and_download()
    
    # 渲染UI
    render_sidebar()
    organism, gene, cell_line, exp_type, analyze = render_main_panel()
    
    if analyze:
        if not gene:
            st.error("⚠️ 请输入或选择一个基因")
            return
        
        # 验证基因名
        is_valid, error_msg = SecurityConfig.validate_gene_name(gene)
        if not is_valid:
            st.error(f"输入验证失败: {error_msg}")
            return
        
        # 清理输入
        gene_clean = SecurityConfig.sanitize_input(gene, 50)
        cell_clean = SecurityConfig.sanitize_input(cell_line, 100) if cell_line else None
        
        # 物种映射
        organism_map = {
            "human": "Homo sapiens",
            "mouse": "Mus musculus",
            "rat": "Rattus norvegicus",
            "cho": "Cricetulus griseus",
            "pig": "Sus scrofa",
            "monkey": "Macaca mulatta"
        }
        organism_clean = organism_map.get(organism, organism)
        
        # 获取API凭证
        email, ncbi_key, error = APIConfig.get_ncbi_credentials()
        if error:
            st.error(error)
            return
        
        # 获取AI API Key（可选）
        ai_key = APIConfig.get_qwen_api_key()
        
        try:
            # 执行评估
            engine = HybridAssessmentEngine(
                email=email,
                ncbi_api_key=ncbi_key,
                ai_api_key=ai_key
            )
            
            with st.spinner("正在进行混合策略评估（核心数据库+文献补充+AI语义分析）..."):
                result = engine.assess(gene_clean, organism_clean, cell_clean, exp_type)
            
            render_results(result)
            
        except Exception as e:
            logger.exception(f"Unhandled error: {e}")
            st.error(f"❌ 系统错误: {str(e)}")  # 显示具体错误信息便于调试

if __name__ == "__main__":
    main()
