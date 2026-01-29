# Save this as: export_engine.py
# Location: C:\Users\Bhoomi\Major Project\08_Output_Integration\export_engine.py

import json
import pandas as pd
from typing import Dict, List
from pathlib import Path
from datetime import datetime


class TestCaseExportEngine:
    """Multi-format test case export engine"""
    
    def __init__(self, test_cases_file: str, validation_file: str, rtm_file: str):
        self.test_cases_file = test_cases_file
        self.validation_file = validation_file
        self.rtm_file = rtm_file
        
        with open(test_cases_file, 'r', encoding='utf-8') as f:
            tc_data = json.load(f)
        
        self.test_cases = tc_data.get('phase1_test_cases', []) + tc_data.get('phase2_test_cases', [])
        self.metadata = tc_data.get('metadata', {})
        
        try:
            self.validation_df = pd.read_excel(validation_file, sheet_name='Validation Results')
        except:
            self.validation_df = pd.DataFrame()
        
        try:
            self.rtm_df = pd.read_excel(rtm_file, sheet_name='RTM Summary')
        except:
            self.rtm_df = pd.DataFrame()
    
    def export_json(self, output_file: str):
        """Export to JSON"""
        export_data = {
            'metadata': {
                'exported_at': datetime.now().isoformat(),
                'total_test_cases': len(self.test_cases),
                'format': 'json',
                'generator': 'AI-Powered Test Case Generation System'
            },
            'test_cases': self.test_cases
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"✅ Exported JSON: {output_file}")
        return output_file
    
    def export_csv(self, output_file: str):
        """Export to CSV"""
        flattened = []
        for tc in self.test_cases:
            flattened.append({
                'Test ID': tc['test_id'],
                'Requirement ID': tc['requirement_id'],
                'Test Type': tc['test_type'],
                'Test Title': tc['test_title'],
                'Description': tc['description'],
                'Priority': tc['priority'],
                'Preconditions': '; '.join(tc['preconditions']),
                'Test Steps': '\n'.join([f"{i+1}. {step}" for i, step in enumerate(tc['test_steps'])]),
                'Expected Result': tc['expected_result'],
                'Test Data': json.dumps(tc['test_data']),
                'Generation Phase': tc['generation_phase'],
                'SRS Section': tc.get('srs_section', '')
            })
        
        df = pd.DataFrame(flattened)
        df.to_csv(output_file, index=False, encoding='utf-8')
        
        print(f"✅ Exported CSV: {output_file}")
        return output_file
    
    def export_testrail(self, output_file: str):
        """Export in TestRail CSV import format"""
        testrail_cases = []
        for tc in self.test_cases:
            testrail_cases.append({
                'Section': tc['requirement_id'],
                'Test Case': tc['test_title'],
                'Priority': 'High' if tc['priority'] == 1 else ('Medium' if tc['priority'] == 2 else 'Low'),
                'Type': tc['test_type'].capitalize(),
                'Preconditions': '\n'.join(tc['preconditions']),
                'Steps': '\n'.join([f"{i+1}. {step}" for i, step in enumerate(tc['test_steps'])]),
                'Expected Result': tc['expected_result'],
                'References': tc.get('srs_section', ''),
                'Custom Fields': json.dumps({
                    'generation_phase': tc['generation_phase'],
                    'test_data': tc['test_data']
                })
            })
        
        df = pd.DataFrame(testrail_cases)
        df.to_csv(output_file, index=False, encoding='utf-8')
        
        print(f"✅ Exported TestRail format: {output_file}")
        return output_file
    
    def export_jira(self, output_file: str):
        """Export in Jira Test Management (Xray) format"""
        jira_cases = []
        for tc in self.test_cases:
            test_steps_markdown = "h3. Preconditions\n"
            test_steps_markdown += "\n".join([f"* {p}" for p in tc['preconditions']])
            test_steps_markdown += "\n\nh3. Test Steps\n"
            test_steps_markdown += "\n".join([f"# {step}" for step in tc['test_steps']])
            test_steps_markdown += "\n\nh3. Expected Result\n"
            test_steps_markdown += tc['expected_result']
            
            jira_cases.append({
                'Issue Type': 'Test',
                'Summary': tc['test_title'],
                'Description': test_steps_markdown,
                'Priority': 'High' if tc['priority'] == 1 else ('Medium' if tc['priority'] == 2 else 'Low'),
                'Components': tc['requirement_id'],
                'Test Type': tc['test_type'].capitalize(),
                'Labels': f"automated,ai-generated,{tc['generation_phase']}",
                'Requirement Link': tc['requirement_id'],
                'Custom Field (Test Data)': json.dumps(tc['test_data'])
            })
        
        df = pd.DataFrame(jira_cases)
        df.to_csv(output_file, index=False, encoding='utf-8')
        
        print(f"✅ Exported Jira format: {output_file}")
        return output_file
    
    def export_all_formats(self, output_dir: str):
        """Export all formats at once"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("\n" + "="*80)
        print(" " * 25 + "MULTI-FORMAT EXPORT")
        print("="*80)
        
        exports = {
            'json': self.export_json(f"{output_dir}/test_cases_{timestamp}.json"),
            'csv': self.export_csv(f"{output_dir}/test_cases_{timestamp}.csv"),
            'testrail': self.export_testrail(f"{output_dir}/test_cases_testrail_{timestamp}.csv"),
            'jira': self.export_jira(f"{output_dir}/test_cases_jira_{timestamp}.csv")
        }
        
        print("\n" + "="*80)
        print("✅ All formats exported successfully!")
        print("="*80)
        
        return exports


def main():
    """Main execution"""
    TEST_CASES_FILE = "../04_AI_powered_TestCaseGeneration/optimized_test_cases_20251017_000535.json"
    VALIDATION_FILE = "../06_Validation_QA/validation_report.xlsx"
    RTM_FILE = "../05_RTM_Generation/rtm_report.xlsx"
    OUTPUT_DIR = "../08_Output_Integration"
    
    exporter = TestCaseExportEngine(TEST_CASES_FILE, VALIDATION_FILE, RTM_FILE)
    exports = exporter.export_all_formats(OUTPUT_DIR)
    
    print(f"\n📁 All exports saved to: {OUTPUT_DIR}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
