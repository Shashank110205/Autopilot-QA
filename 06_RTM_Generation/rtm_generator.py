import json
import duckdb
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


# ============================================
# STEP 1: DATABASE SETUP & SCHEMA CREATION
# ============================================

class RTMDatabase:
    """Requirements Traceability Matrix Database Manager"""
    
    def __init__(self, db_path: str = "rtm_database.duckdb"):
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)
        print(f"✅ Connected to DuckDB: {db_path}")
        self._create_schema()
    
    def _create_schema(self):
        """Create database schema for RTM"""
        
        print("\n" + "="*80)
        print("STEP 1: Creating Database Schema")
        print("="*80)
        
        # Requirements table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS requirements (
                id VARCHAR PRIMARY KEY,
                title VARCHAR,
                content TEXT,
                type VARCHAR,
                priority INTEGER,
                domain VARCHAR,
                dependencies JSON,
                rationale TEXT,
                created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        print("✅ Created table: requirements")
        
        # Test cases table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS test_cases (
                id VARCHAR PRIMARY KEY,
                requirement_id VARCHAR,
                test_type VARCHAR,
                test_title VARCHAR,
                description TEXT,
                priority INTEGER,
                confidence_score FLOAT DEFAULT 1.0,
                validation_status BOOLEAN DEFAULT TRUE,
                test_steps JSON,
                preconditions JSON,
                test_data JSON,
                expected_result TEXT,
                generation_phase VARCHAR,
                srs_section VARCHAR,
                depends_on JSON,
                created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        print("✅ Created table: test_cases")
        
        # Traceability mapping table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS rtm_mapping (
                requirement_id VARCHAR,
                test_case_id VARCHAR,
                mapping_confidence FLOAT DEFAULT 1.0,
                mapping_type VARCHAR,
                created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (requirement_id, test_case_id)
            )
        """)
        print("✅ Created table: rtm_mapping")
        
        # Coverage metrics table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS coverage_metrics (
                metric_name VARCHAR PRIMARY KEY,
                metric_value FLOAT,
                target_value FLOAT,
                status VARCHAR,
                calculated_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        print("✅ Created table: coverage_metrics")
        
        # Gap analysis table - FIXED: Composite primary key
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS gap_analysis (
                requirement_id VARCHAR,
                gap_type VARCHAR,
                severity VARCHAR,
                recommendation TEXT,
                identified_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (requirement_id, gap_type)
            )
        """)
        print("✅ Created table: gap_analysis")
        
        print("\n✅ Database schema created successfully!\n")
    
    def close(self):
        """Close database connection"""
        self.conn.close()


# ============================================
# STEP 2: DATA INGESTION ENGINE
# ============================================

class RTMDataIngestion:
    """Ingests requirements and test cases into RTM database"""
    
    def __init__(self, db: RTMDatabase):
        self.db = db
        self.conn = db.conn
    
    def ingest_requirements(self, requirements_json: str):
        """Ingest requirements from chunked JSON file"""
        
        print("="*80)
        print("STEP 2A: Ingesting Requirements")
        print("="*80)
        
        with open(requirements_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        requirements = []
        for chunk in data['chunks']:
            for req in chunk['requirements']:
                requirements.append({
                    'id': req['id'],
                    'title': req['title'],
                    'content': req['description'],
                    'type': 'functional' if req['id'].startswith('FR') else 'non_functional',
                    'priority': 1 if req.get('priority') == 'High' else (2 if req.get('priority') == 'Medium' else 3),
                    'domain': data['domain_classification']['primary_domain'],
                    'dependencies': json.dumps(req.get('dependencies', [])),
                    'rationale': req.get('rationale', ''),
                    'created_timestamp': datetime.now()
                })
        
        # Deduplicate requirements
        seen_ids = set()
        unique_requirements = []
        for req in requirements:
            if req['id'] not in seen_ids:
                unique_requirements.append(req)
                seen_ids.add(req['id'])
        
        # Insert into database
        df = pd.DataFrame(unique_requirements)
        self.conn.execute("DELETE FROM requirements")
        self.conn.execute("INSERT INTO requirements SELECT * FROM df")
        
        count = self.conn.execute("SELECT COUNT(*) FROM requirements").fetchone()[0]
        print(f"✅ Ingested {count} unique requirements")
        
        sample = self.conn.execute("""
            SELECT id, title, type, priority 
            FROM requirements 
            LIMIT 5
        """).fetchdf()
        print("\nSample Requirements:")
        print(sample.to_string(index=False))
        print()
        
        return count
    
    def ingest_test_cases(self, test_cases_json: str):
        """Ingest test cases from generated JSON file"""
        
        print("="*80)
        print("STEP 2B: Ingesting Test Cases")
        print("="*80)
        
        with open(test_cases_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        all_test_cases = data.get('phase1_test_cases', []) + data.get('phase2_test_cases', [])
        
        test_cases = []
        for tc in all_test_cases:
            test_cases.append({
                'id': tc['test_id'],
                'requirement_id': tc['requirement_id'],
                'test_type': tc['test_type'],
                'test_title': tc['test_title'],
                'description': tc.get('description', ''),
                'priority': 1 if tc['priority'] == 'High' else (2 if tc['priority'] == 'Medium' else 3),
                'confidence_score': 1.0,
                'validation_status': True,
                'test_steps': json.dumps(tc['test_steps']),
                'preconditions': json.dumps(tc['preconditions']),
                'test_data': json.dumps(tc['test_data']),
                'expected_result': tc['expected_result'],
                'generation_phase': tc['generation_phase'],
                'srs_section': tc.get('srs_section', ''),
                'depends_on': json.dumps(tc.get('depends_on', [])),
                'created_timestamp': datetime.now()
            })
        
        df = pd.DataFrame(test_cases)
        self.conn.execute("DELETE FROM test_cases")
        self.conn.execute("INSERT INTO test_cases SELECT * FROM df")
        
        count = self.conn.execute("SELECT COUNT(*) FROM test_cases").fetchone()[0]
        print(f"✅ Ingested {count} test cases")
        
        sample = self.conn.execute("""
            SELECT id, requirement_id, test_type, priority 
            FROM test_cases 
            LIMIT 5
        """).fetchdf()
        print("\nSample Test Cases:")
        print(sample.to_string(index=False))
        print()
        
        return count
    
    def create_traceability_mappings(self):
        """Create bi-directional traceability mappings"""
        
        print("="*80)
        print("STEP 2C: Creating Traceability Mappings")
        print("="*80)
        
        self.conn.execute("""
            INSERT INTO rtm_mapping (requirement_id, test_case_id, mapping_confidence, mapping_type)
            SELECT 
                tc.requirement_id,
                tc.id AS test_case_id,
                tc.confidence_score AS mapping_confidence,
                'direct' AS mapping_type
            FROM test_cases tc
        """)
        
        count = self.conn.execute("SELECT COUNT(*) FROM rtm_mapping").fetchone()[0]
        print(f"✅ Created {count} traceability mappings")
        
        sample = self.conn.execute("""
            SELECT requirement_id, test_case_id, mapping_confidence, mapping_type
            FROM rtm_mapping
            LIMIT 5
        """).fetchdf()
        print("\nSample Mappings:")
        print(sample.to_string(index=False))
        print()
        
        return count


# ============================================
# STEP 3: COVERAGE ANALYSIS ENGINE
# ============================================

class CoverageAnalyzer:
    """Analyzes test coverage metrics and identifies gaps"""
    
    def __init__(self, db: RTMDatabase):
        self.db = db
        self.conn = db.conn
    
    def calculate_functional_coverage(self) -> float:
        """Calculate % of requirements with test cases (Target: 95%)"""
        
        result = self.conn.execute("""
            WITH req_coverage AS (
                SELECT 
                    r.id,
                    COUNT(DISTINCT tc.id) as test_count
                FROM requirements r
                LEFT JOIN test_cases tc ON r.id = tc.requirement_id
                GROUP BY r.id
            )
            SELECT 
                COUNT(*) FILTER (WHERE test_count > 0) * 100.0 / COUNT(*) as coverage_pct
            FROM req_coverage
        """).fetchone()
        
        return result[0] if result else 0.0
    
    def calculate_edge_case_coverage(self) -> float:
        """Calculate % of requirements with edge/negative test cases (Target: 80%)"""
        
        result = self.conn.execute("""
            WITH edge_coverage AS (
                SELECT 
                    r.id,
                    COUNT(DISTINCT tc.id) FILTER (WHERE tc.test_type IN ('edge', 'negative')) as edge_count
                FROM requirements r
                LEFT JOIN test_cases tc ON r.id = tc.requirement_id
                GROUP BY r.id
            )
            SELECT 
                COUNT(*) FILTER (WHERE edge_count > 0) * 100.0 / COUNT(*) as coverage_pct
            FROM edge_coverage
        """).fetchone()
        
        return result[0] if result else 0.0
    
    def calculate_integration_coverage(self) -> float:
        """Calculate % of requirements with dependencies having integration tests (Target: 85%)"""
        
        result = self.conn.execute("""
            WITH integration_coverage AS (
                SELECT 
                    r.id,
                    r.dependencies,
                    COUNT(DISTINCT tc.id) FILTER (WHERE tc.test_type = 'integration') as integration_count
                FROM requirements r
                LEFT JOIN test_cases tc ON r.id = tc.requirement_id
                WHERE r.dependencies != '[]' AND r.dependencies IS NOT NULL
                GROUP BY r.id, r.dependencies
            )
            SELECT 
                COUNT(*) FILTER (WHERE integration_count > 0) * 100.0 / NULLIF(COUNT(*), 0) as coverage_pct
            FROM integration_coverage
        """).fetchone()
        
        return result[0] if result else 100.0
    
    def calculate_test_type_distribution(self) -> pd.DataFrame:
        """Calculate distribution of test types"""
        
        result = self.conn.execute("""
            SELECT 
                test_type,
                COUNT(*) as count,
                COUNT(*) * 100.0 / SUM(COUNT(*)) OVER() as percentage
            FROM test_cases
            GROUP BY test_type
            ORDER BY count DESC
        """).fetchdf()
        
        return result
    
    def calculate_priority_coverage(self) -> pd.DataFrame:
        """Calculate coverage by requirement priority"""
        
        result = self.conn.execute("""
            SELECT 
                r.priority,
                COUNT(DISTINCT r.id) as total_requirements,
                COUNT(DISTINCT tc.id) as total_tests,
                COUNT(DISTINCT tc.id) * 1.0 / COUNT(DISTINCT r.id) as avg_tests_per_req,
                COUNT(DISTINCT r.id) FILTER (WHERE tc.id IS NOT NULL) * 100.0 / COUNT(DISTINCT r.id) as coverage_pct
            FROM requirements r
            LEFT JOIN test_cases tc ON r.id = tc.requirement_id
            GROUP BY r.priority
            ORDER BY r.priority
        """).fetchdf()
        
        return result
    
    def run_full_coverage_analysis(self):
        """Run complete coverage analysis"""
        
        print("="*80)
        print("STEP 3: Coverage Analysis")
        print("="*80)
        
        functional_coverage = self.calculate_functional_coverage()
        edge_case_coverage = self.calculate_edge_case_coverage()
        integration_coverage = self.calculate_integration_coverage()
        
        metrics = [
            ('functional_coverage', functional_coverage, 95.0),
            ('edge_case_coverage', edge_case_coverage, 80.0),
            ('integration_coverage', integration_coverage, 85.0)
        ]
        
        self.conn.execute("DELETE FROM coverage_metrics")
        
        for metric_name, metric_value, target_value in metrics:
            status = 'achieved' if metric_value >= target_value else 'below_target'
            if metric_value > target_value + 5:
                status = 'exceeds_target'
            
            self.conn.execute("""
                INSERT INTO coverage_metrics (metric_name, metric_value, target_value, status)
                VALUES (?, ?, ?, ?)
            """, [metric_name, metric_value, target_value, status])
        
        print(f"\n📊 Coverage Metrics:")
        print(f"{'='*80}")
        print(f"{'Metric':<30} | {'Actual':>10} | {'Target':>10} | {'Status':>15}")
        print(f"{'-'*80}")
        
        for metric_name, metric_value, target_value in metrics:
            status = '✅ Achieved' if metric_value >= target_value else '❌ Below Target'
            print(f"{metric_name.replace('_', ' ').title():<30} | {metric_value:>9.1f}% | {target_value:>9.1f}% | {status:>15}")
        
        print(f"{'='*80}\n")
        
        print("📊 Test Type Distribution:")
        print("="*80)
        type_dist = self.calculate_test_type_distribution()
        print(type_dist.to_string(index=False))
        print()
        
        print("📊 Coverage by Priority:")
        print("="*80)
        priority_cov = self.calculate_priority_coverage()
        priority_cov['priority'] = priority_cov['priority'].map({1: 'High', 2: 'Medium', 3: 'Low'})
        print(priority_cov.to_string(index=False))
        print()
        
        return {
            'functional_coverage': functional_coverage,
            'edge_case_coverage': edge_case_coverage,
            'integration_coverage': integration_coverage
        }


# ============================================
# STEP 4: GAP ANALYSIS ENGINE (FIXED)
# ============================================

class GapAnalyzer:
    """Identifies gaps in test coverage"""
    
    def __init__(self, db: RTMDatabase):
        self.db = db
        self.conn = db.conn
    
    def identify_requirements_without_tests(self) -> pd.DataFrame:
        """Find requirements with NO test cases"""
        
        result = self.conn.execute("""
            SELECT 
                r.id,
                r.title,
                r.type,
                r.priority
            FROM requirements r
            LEFT JOIN test_cases tc ON r.id = tc.requirement_id
            WHERE tc.id IS NULL
            ORDER BY r.priority
        """).fetchdf()
        
        return result
    
    def identify_insufficient_coverage(self, min_tests: int = 3) -> pd.DataFrame:
        """Find requirements with insufficient test cases"""
        
        result = self.conn.execute(f"""
            SELECT 
                r.id,
                r.title,
                r.priority,
                COUNT(tc.id) as test_count
            FROM requirements r
            LEFT JOIN test_cases tc ON r.id = tc.requirement_id
            GROUP BY r.id, r.title, r.priority
            HAVING COUNT(tc.id) < {min_tests} AND COUNT(tc.id) > 0
            ORDER BY test_count ASC, r.priority
        """).fetchdf()
        
        return result
    
    def identify_missing_test_types(self) -> pd.DataFrame:
        """Find requirements missing critical test types"""
        
        result = self.conn.execute("""
            WITH req_test_types AS (
                SELECT 
                    r.id,
                    r.title,
                    r.priority,
                    ARRAY_AGG(DISTINCT tc.test_type) as test_types
                FROM requirements r
                LEFT JOIN test_cases tc ON r.id = tc.requirement_id
                WHERE tc.id IS NOT NULL
                GROUP BY r.id, r.title, r.priority
            )
            SELECT 
                id,
                title,
                priority,
                test_types,
                CASE 
                    WHEN NOT list_contains(test_types, 'positive') THEN 'missing_positive'
                    WHEN NOT list_contains(test_types, 'negative') THEN 'missing_negative'
                    WHEN NOT list_contains(test_types, 'edge') THEN 'missing_edge'
                    ELSE 'ok'
                END as gap_type
            FROM req_test_types
            WHERE 
                NOT list_contains(test_types, 'positive') OR
                NOT list_contains(test_types, 'negative') OR
                NOT list_contains(test_types, 'edge')
            ORDER BY priority
        """).fetchdf()
        
        return result
    
    def run_gap_analysis(self):
        """Run complete gap analysis - FIXED with UPSERT logic"""
        
        print("="*80)
        print("STEP 4: Gap Analysis")
        print("="*80)
        
        self.conn.execute("DELETE FROM gap_analysis")
        
        # 1. Requirements without tests
        no_tests = self.identify_requirements_without_tests()
        if not no_tests.empty:
            print(f"\n⚠️ Requirements with NO test cases: {len(no_tests)}")
            print(no_tests.to_string(index=False))
            
            for _, row in no_tests.iterrows():
                severity = 'critical' if row['priority'] == 1 else ('high' if row['priority'] == 2 else 'medium')
                self.conn.execute("""
                    INSERT INTO gap_analysis (requirement_id, gap_type, severity, recommendation)
                    VALUES (?, 'no_tests', ?, 'Generate test cases for this requirement')
                    ON CONFLICT (requirement_id, gap_type) DO NOTHING
                """, [row['id'], severity])
        else:
            print("\n✅ All requirements have test cases!")
        
        # 2. Insufficient coverage
        insufficient = self.identify_insufficient_coverage(min_tests=3)
        if not insufficient.empty:
            print(f"\n⚠️ Requirements with < 3 test cases: {len(insufficient)}")
            print(insufficient.to_string(index=False))
            
            for _, row in insufficient.iterrows():
                severity = 'high' if row['priority'] == 1 else 'medium'
                self.conn.execute("""
                    INSERT INTO gap_analysis (requirement_id, gap_type, severity, recommendation)
                    VALUES (?, 'insufficient_coverage', ?, ?)
                    ON CONFLICT (requirement_id, gap_type) DO NOTHING
                """, [row['id'], severity, f"Add more test cases (current: {row['test_count']}, target: 3+)"])
        else:
            print("\n✅ All requirements have sufficient test cases!")
        
        # 3. Missing test types - FIXED: Insert one gap per missing type
        missing_types = self.identify_missing_test_types()
        if not missing_types.empty:
            print(f"\n⚠️ Requirements missing critical test types: {len(missing_types)}")
            print(missing_types[['id', 'title', 'priority', 'gap_type']].to_string(index=False))
            
            for _, row in missing_types.iterrows():
                gap_type = row['gap_type']  # 'missing_positive', 'missing_negative', or 'missing_edge'
                recommendation = f"Add {gap_type.replace('missing_', '')} test case"
                
                self.conn.execute("""
                    INSERT INTO gap_analysis (requirement_id, gap_type, severity, recommendation)
                    VALUES (?, ?, 'medium', ?)
                    ON CONFLICT (requirement_id, gap_type) DO NOTHING
                """, [row['id'], gap_type, recommendation])
        else:
            print("\n✅ All requirements have critical test types!")
        
        # Summary
        gap_summary = self.conn.execute("""
            SELECT 
                gap_type,
                severity,
                COUNT(*) as count
            FROM gap_analysis
            GROUP BY gap_type, severity
            ORDER BY severity, gap_type
        """).fetchdf()
        
        if not gap_summary.empty:
            print(f"\n📊 Gap Analysis Summary:")
            print("="*80)
            print(gap_summary.to_string(index=False))
            print()
        
        return gap_summary


# ============================================
# STEP 5: RTM REPORT GENERATOR
# ============================================

class RTMReportGenerator:
    """Generates comprehensive RTM reports"""
    
    def __init__(self, db: RTMDatabase):
        self.db = db
        self.conn = db.conn
    
    def generate_full_rtm(self) -> pd.DataFrame:
        """Generate complete bi-directional RTM"""
        
        rtm = self.conn.execute("""
            SELECT 
                r.id as requirement_id,
                r.title as requirement_title,
                r.type as requirement_type,
                r.priority as requirement_priority,
                tc.id as test_case_id,
                tc.test_type,
                tc.test_title,
                tc.priority as test_priority,
                tc.generation_phase,
                m.mapping_confidence,
                m.mapping_type
            FROM requirements r
            LEFT JOIN rtm_mapping m ON r.id = m.requirement_id
            LEFT JOIN test_cases tc ON m.test_case_id = tc.id
            ORDER BY r.id, tc.test_type
        """).fetchdf()
        
        return rtm
    
    def generate_summary_rtm(self) -> pd.DataFrame:
        """Generate summary RTM"""
        
        summary = self.conn.execute("""
            SELECT 
                r.id,
                r.title,
                r.type,
                r.priority,
                COUNT(tc.id) as total_tests,
                COUNT(tc.id) FILTER (WHERE tc.test_type = 'positive') as positive_tests,
                COUNT(tc.id) FILTER (WHERE tc.test_type = 'negative') as negative_tests,
                COUNT(tc.id) FILTER (WHERE tc.test_type = 'edge') as edge_tests,
                COUNT(tc.id) FILTER (WHERE tc.test_type = 'integration') as integration_tests,
                COUNT(tc.id) FILTER (WHERE tc.test_type = 'performance') as performance_tests,
                COUNT(tc.id) FILTER (WHERE tc.test_type = 'security') as security_tests,
                CASE 
                    WHEN COUNT(tc.id) = 0 THEN '❌ No Tests'
                    WHEN COUNT(tc.id) < 3 THEN '⚠️ Low Coverage'
                    WHEN COUNT(tc.id) >= 3 AND COUNT(tc.id) < 6 THEN '✅ Good Coverage'
                    ELSE '✅ Excellent Coverage'
                END as coverage_status
            FROM requirements r
            LEFT JOIN test_cases tc ON r.id = tc.requirement_id
            GROUP BY r.id, r.title, r.type, r.priority
            ORDER BY r.id
        """).fetchdf()
        
        return summary
    
    def export_to_excel(self, output_file: str):
        """Export RTM to Excel"""
        
        print("="*80)
        print("STEP 5: Generating RTM Reports")
        print("="*80)
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Sheet 1: Summary RTM
            summary = self.generate_summary_rtm()
            summary.to_excel(writer, sheet_name='RTM Summary', index=False)
            print(f"✅ Sheet 1: RTM Summary ({len(summary)} requirements)")
            
            # Sheet 2: Full RTM
            full_rtm = self.generate_full_rtm()
            full_rtm.to_excel(writer, sheet_name='Full RTM', index=False)
            print(f"✅ Sheet 2: Full RTM ({len(full_rtm)} mappings)")
            
            # Sheet 3: Coverage Metrics
            metrics = self.conn.execute("SELECT * FROM coverage_metrics").fetchdf()
            metrics.to_excel(writer, sheet_name='Coverage Metrics', index=False)
            print(f"✅ Sheet 3: Coverage Metrics ({len(metrics)} metrics)")
            
            # Sheet 4: Gap Analysis
            gaps = self.conn.execute("SELECT * FROM gap_analysis ORDER BY severity, requirement_id").fetchdf()
            if not gaps.empty:
                gaps.to_excel(writer, sheet_name='Gap Analysis', index=False)
                print(f"✅ Sheet 4: Gap Analysis ({len(gaps)} gaps)")
            
            # Sheet 5: Test Type Distribution
            type_dist = self.conn.execute("""
                SELECT test_type, COUNT(*) as count
                FROM test_cases
                GROUP BY test_type
                ORDER BY count DESC
            """).fetchdf()
            type_dist.to_excel(writer, sheet_name='Test Type Distribution', index=False)
            print(f"✅ Sheet 5: Test Type Distribution")
            
            # Sheet 6: Requirements without tests
            no_tests = self.conn.execute("""
                SELECT r.id, r.title, r.type, r.priority
                FROM requirements r
                LEFT JOIN test_cases tc ON r.id = tc.requirement_id
                WHERE tc.id IS NULL
            """).fetchdf()
            if not no_tests.empty:
                no_tests.to_excel(writer, sheet_name='Requirements Without Tests', index=False)
                print(f"✅ Sheet 6: Requirements Without Tests ({len(no_tests)} requirements)")
        
        print(f"\n✅ RTM Excel report saved: {output_file}\n")
    
    def generate_html_dashboard(self, output_file: str):
        """Generate HTML dashboard"""
        
        summary = self.generate_summary_rtm()
        metrics = self.conn.execute("SELECT * FROM coverage_metrics").fetchdf()
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Requirements Traceability Matrix Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .header {{ background: #2c3e50; color: white; padding: 20px; border-radius: 5px; }}
        .metric-card {{ background: white; padding: 20px; margin: 10px 0; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .metric-value {{ font-size: 36px; font-weight: bold; color: #3498db; }}
        .metric-label {{ font-size: 14px; color: #7f8c8d; }}
        table {{ width: 100%; border-collapse: collapse; background: white; margin: 20px 0; }}
        th {{ background: #3498db; color: white; padding: 12px; text-align: left; }}
        td {{ padding: 10px; border-bottom: 1px solid #ddd; }}
        tr:hover {{ background: #f8f9fa; }}
        .status-good {{ color: #27ae60; font-weight: bold; }}
        .status-warning {{ color: #f39c12; font-weight: bold; }}
        .status-bad {{ color: #e74c3c; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Requirements Traceability Matrix (RTM) Dashboard</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <h2>Coverage Metrics</h2>
    <div style="display: flex; gap: 20px;">
"""
        
        for _, row in metrics.iterrows():
            status_class = 'status-good' if row['status'] == 'achieved' else 'status-bad'
            html += f"""
        <div class="metric-card" style="flex: 1;">
            <div class="metric-label">{row['metric_name'].replace('_', ' ').title()}</div>
            <div class="metric-value {status_class}">{row['metric_value']:.1f}%</div>
            <div class="metric-label">Target: {row['target_value']:.0f}%</div>
        </div>
"""
        
        html += """
    </div>
    
    <h2>Requirements Coverage Summary (Top 20)</h2>
    <table>
        <tr>
            <th>ID</th>
            <th>Title</th>
            <th>Type</th>
            <th>Priority</th>
            <th>Total Tests</th>
            <th>Status</th>
        </tr>
"""
        
        for _, row in summary.head(20).iterrows():
            html += f"""
        <tr>
            <td>{row['id']}</td>
            <td>{row['title'][:60]}...</td>
            <td>{row['type']}</td>
            <td>{row['priority']}</td>
            <td>{row['total_tests']}</td>
            <td>{row['coverage_status']}</td>
        </tr>
"""
        
        html += """
    </table>
</body>
</html>
"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"✅ HTML dashboard saved: {output_file}\n")


# ============================================
# STEP 6: MAIN ORCHESTRATOR
# ============================================

def main():
    """Main execution"""
    
    print("\n" + "="*80)
    print(" " * 20 + "RTM GENERATION PIPELINE")
    print("="*80)
    print("\nStarting Requirements Traceability Matrix generation...\n")
    
    REQUIREMENTS_FILE = "../03_Chunking_Domain_Understanding/chunked_requirements_with_domain.json"
    TEST_CASES_FILE = "../04_AI_powered_TestCaseGeneration/optimized_test_cases_20251017_000535.json"
    DB_PATH = "../05_RTM_Generation/rtm_database.duckdb"
    EXCEL_OUTPUT = "../05_RTM_Generation/rtm_report.xlsx"
    HTML_OUTPUT = "../05_RTM_Generation/rtm_dashboard.html"
    
    Path("../05_RTM_Generation").mkdir(parents=True, exist_ok=True)
    
    try:
        db = RTMDatabase(DB_PATH)
        
        ingestion = RTMDataIngestion(db)
        req_count = ingestion.ingest_requirements(REQUIREMENTS_FILE)
        tc_count = ingestion.ingest_test_cases(TEST_CASES_FILE)
        mapping_count = ingestion.create_traceability_mappings()
        
        coverage = CoverageAnalyzer(db)
        metrics = coverage.run_full_coverage_analysis()
        
        gap_analyzer = GapAnalyzer(db)
        gaps = gap_analyzer.run_gap_analysis()
        
        report_gen = RTMReportGenerator(db)
        report_gen.export_to_excel(EXCEL_OUTPUT)
        report_gen.generate_html_dashboard(HTML_OUTPUT)
        
        print("="*80)
        print(" " * 25 + "✅ RTM GENERATION COMPLETE!")
        print("="*80)
        print(f"\n📊 Summary:")
        print(f"  • Requirements: {req_count}")
        print(f"  • Test Cases: {tc_count}")
        print(f"  • Traceability Mappings: {mapping_count}")
        print(f"\n📈 Coverage:")
        print(f"  • Functional Coverage: {metrics['functional_coverage']:.1f}% (Target: 95%)")
        print(f"  • Edge Case Coverage: {metrics['edge_case_coverage']:.1f}% (Target: 80%)")
        print(f"  • Integration Coverage: {metrics['integration_coverage']:.1f}% (Target: 85%)")
        print(f"\n📁 Output Files:")
        print(f"  • Database: {DB_PATH}")
        print(f"  • Excel Report: {EXCEL_OUTPUT}")
        print(f"  • HTML Dashboard: {HTML_OUTPUT}")
        print(f"\n{'='*80}\n")
        
        db.close()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
