import json
import pandas as pd
from datetime import datetime
from typing import Dict, List
from collections import defaultdict
from pathlib import Path


# ============================================
# MOCK FEEDBACK COLLECTION SYSTEM
# ============================================

class FeedbackCollector:
    """
    Mock feedback collection system for demonstration purposes.
    In production, this would integrate with UI/API for real user feedback.
    """
    
    def __init__(self, feedback_db_path: str = "feedback_database.json"):
        self.feedback_db_path = feedback_db_path
        self.feedback_data = self._load_feedback()
    
    def _load_feedback(self) -> Dict:
        """Load existing feedback data"""
        if Path(self.feedback_db_path).exists():
            with open(self.feedback_db_path, 'r') as f:
                return json.load(f)
        return {
            'test_case_feedback': [],
            'requirement_patterns': {},
            'common_issues': [],
            'improvement_suggestions': []
        }
    
    def _save_feedback(self):
        """Save feedback data"""
        with open(self.feedback_db_path, 'w') as f:
            json.dump(self.feedback_data, f, indent=2)
    
    def collect_feedback(self, test_case_id: str, feedback_type: str, 
                        feedback_data: Dict, user_id: str = "system"):
        """
        Collect feedback on a test case.
        
        Feedback types:
        - 'approved': Test case accepted as-is
        - 'rejected': Test case rejected
        - 'modified': Test case modified by user
        - 'flagged': Test case flagged for review
        """
        
        feedback_entry = {
            'test_case_id': test_case_id,
            'feedback_type': feedback_type,
            'feedback_data': feedback_data,
            'user_id': user_id,
            'timestamp': datetime.now().isoformat()
        }
        
        self.feedback_data['test_case_feedback'].append(feedback_entry)
        self._save_feedback()
        
        print(f"✅ Feedback collected: {feedback_type} for {test_case_id}")
    
    def get_feedback_summary(self) -> Dict:
        """Get summary of collected feedback"""
        
        feedback = self.feedback_data['test_case_feedback']
        
        if not feedback:
            return {
                'total_feedback': 0,
                'approved': 0,
                'rejected': 0,
                'modified': 0,
                'flagged': 0,
                'approval_rate': 0.0
            }
        
        type_counts = defaultdict(int)
        for entry in feedback:
            type_counts[entry['feedback_type']] += 1
        
        total = len(feedback)
        approval_rate = (type_counts['approved'] / total * 100) if total > 0 else 0
        
        return {
            'total_feedback': total,
            'approved': type_counts['approved'],
            'rejected': type_counts['rejected'],
            'modified': type_counts['modified'],
            'flagged': type_counts['flagged'],
            'approval_rate': approval_rate
        }


# ============================================
# RULE-BASED IMPROVEMENT ENGINE
# ============================================

class RuleBasedImprovementEngine:
    """
    Simple rule-based improvement without ML fine-tuning.
    Demonstrates feedback-driven improvement concept for academic purposes.
    """
    
    def __init__(self, feedback_collector: FeedbackCollector):
        self.feedback_collector = feedback_collector
        self.improvement_rules = []
    
    def analyze_feedback_patterns(self) -> List[Dict]:
        """Analyze feedback to identify improvement patterns"""
        
        feedback = self.feedback_collector.feedback_data['test_case_feedback']
        
        if not feedback:
            return []
        
        patterns = []
        
        # Pattern 1: Identify commonly rejected test types
        rejected_tests = [f for f in feedback if f['feedback_type'] == 'rejected']
        if rejected_tests:
            rejection_reasons = defaultdict(int)
            for test in rejected_tests:
                reason = test['feedback_data'].get('reason', 'unknown')
                rejection_reasons[reason] += 1
            
            for reason, count in rejection_reasons.items():
                if count >= 3:  # Threshold: 3+ rejections for same reason
                    patterns.append({
                        'pattern_type': 'common_rejection',
                        'reason': reason,
                        'count': count,
                        'improvement_rule': f"Enhance validation for: {reason}"
                    })
        
        # Pattern 2: Identify successful modifications
        modified_tests = [f for f in feedback if f['feedback_type'] == 'modified']
        if modified_tests:
            modification_types = defaultdict(int)
            for test in modified_tests:
                mod_type = test['feedback_data'].get('modification_type', 'unknown')
                modification_types[mod_type] += 1
            
            for mod_type, count in modification_types.items():
                if count >= 3:
                    patterns.append({
                        'pattern_type': 'common_modification',
                        'modification_type': mod_type,
                        'count': count,
                        'improvement_rule': f"Improve generation for: {mod_type}"
                    })
        
        return patterns
    
    def generate_improvement_recommendations(self) -> List[str]:
        """Generate improvement recommendations based on feedback"""
        
        patterns = self.analyze_feedback_patterns()
        recommendations = []
        
        summary = self.feedback_collector.get_feedback_summary()
        approval_rate = summary['approval_rate']
        
        # Recommendation based on approval rate
        if approval_rate < 70:
            recommendations.append(
                "⚠️ Low approval rate (<70%) - Review validation rules and prompt engineering"
            )
        elif approval_rate < 85:
            recommendations.append(
                "⚠️ Moderate approval rate (70-85%) - Fine-tune prompts for specific requirement types"
            )
        else:
            recommendations.append(
                "✅ High approval rate (>85%) - System performing well, focus on edge cases"
            )
        
        # Pattern-based recommendations
        for pattern in patterns:
            if pattern['pattern_type'] == 'common_rejection':
                recommendations.append(
                    f"🔧 Add validation rule: {pattern['improvement_rule']} (rejected {pattern['count']} times)"
                )
            elif pattern['pattern_type'] == 'common_modification':
                recommendations.append(
                    f"🔧 Update generation logic: {pattern['improvement_rule']} (modified {pattern['count']} times)"
                )
        
        return recommendations


# ============================================
# ACTIVE LEARNING SIMULATOR
# ============================================

class ActiveLearningSampler:
    """
    Simulates active learning by identifying test cases that need review.
    In production, this would prioritize uncertain cases for user feedback.
    """
    
    def __init__(self, validation_results: List[Dict]):
        self.validation_results = validation_results
    
    def uncertainty_sampling(self, threshold_low: float = 0.4, 
                           threshold_high: float = 0.7) -> List[Dict]:
        """
        Select test cases with confidence scores between thresholds.
        These are uncertain cases that would benefit most from human feedback.
        """
        
        uncertain_cases = []
        
        for result in self.validation_results:
            confidence = result.get('confidence', {}).get('overall_score', 0)
            
            if threshold_low <= confidence <= threshold_high:
                uncertain_cases.append({
                    'test_id': result['test_id'],
                    'requirement_id': result['requirement_id'],
                    'confidence_score': confidence,
                    'reason': 'Uncertain prediction - would benefit from feedback'
                })
        
        # Sort by confidence (lowest first - most uncertain)
        uncertain_cases.sort(key=lambda x: x['confidence_score'])
        
        return uncertain_cases
    
    def representative_sampling(self, n_samples: int = 20) -> List[Dict]:
        """
        Select diverse representative samples covering different requirements.
        Ensures feedback collection covers various domains.
        """
        
        # Group by requirement
        req_groups = defaultdict(list)
        for result in self.validation_results:
            req_id = result['requirement_id']
            req_groups[req_id].append(result)
        
        # Sample from each requirement
        samples = []
        per_req_samples = max(1, n_samples // len(req_groups))
        
        for req_id, tests in req_groups.items():
            # Sort by confidence variance (get diverse confidence levels)
            sorted_tests = sorted(tests, key=lambda x: x.get('confidence', {}).get('overall_score', 0))
            
            # Take samples from low, medium, high confidence
            if len(sorted_tests) >= 3:
                samples.append(sorted_tests[0])  # Low confidence
                samples.append(sorted_tests[len(sorted_tests)//2])  # Medium
                samples.append(sorted_tests[-1])  # High confidence
            else:
                samples.extend(sorted_tests[:per_req_samples])
        
        return samples[:n_samples]


# ============================================
# MOCK FEEDBACK LOOP ORCHESTRATOR
# ============================================

class MockFeedbackLoop:
    """
    Demonstrates feedback loop concept without real user interaction.
    Generates mock feedback for demonstration purposes.
    """
    
    def __init__(self):
        self.feedback_collector = FeedbackCollector()
        self.improvement_engine = RuleBasedImprovementEngine(self.feedback_collector)
    
    def simulate_feedback_collection(self, validation_results: List[Dict], 
                                     n_samples: int = 50):
        """
        Simulate collecting feedback on test cases.
        In production, this would be real user feedback via UI.
        """
        
        print("\n" + "="*80)
        print(" " * 20 + "SIMULATING FEEDBACK COLLECTION")
        print("="*80)
        print(f"\nSimulating feedback for {n_samples} test cases...")
        
        # Use active learning to select samples
        sampler = ActiveLearningSampler(validation_results)
        uncertain_samples = sampler.uncertainty_sampling()[:n_samples]
        
        # Simulate feedback based on confidence scores
        for sample in uncertain_samples:
            confidence = sample['confidence_score']
            
            # Simulate realistic feedback distribution
            if confidence >= 0.6:
                # High confidence -> likely approved
                feedback_type = 'approved'
                feedback_data = {'reason': 'Test case meets quality standards'}
            elif confidence >= 0.5:
                # Medium confidence -> might be modified
                feedback_type = 'modified'
                feedback_data = {
                    'modification_type': 'test_steps_refinement',
                    'reason': 'Test steps need more specificity'
                }
            else:
                # Low confidence -> likely rejected
                feedback_type = 'rejected'
                feedback_data = {'reason': 'Insufficient test coverage'}
            
            self.feedback_collector.collect_feedback(
                test_case_id=sample['test_id'],
                feedback_type=feedback_type,
                feedback_data=feedback_data,
                user_id='simulator'
            )
        
        print(f"✅ Simulated feedback collection complete!")
    
    def generate_improvement_report(self, output_file: str):
        """Generate feedback analysis and improvement report"""
        
        print("\n" + "="*80)
        print(" " * 20 + "GENERATING IMPROVEMENT REPORT")
        print("="*80)
        
        # Get feedback summary
        summary = self.feedback_collector.get_feedback_summary()
        
        # Analyze patterns
        patterns = self.improvement_engine.analyze_feedback_patterns()
        
        # Generate recommendations
        recommendations = self.improvement_engine.generate_improvement_recommendations()
        
        # Display console summary
        print(f"\n📊 Feedback Summary:")
        print(f"  Total Feedback: {summary['total_feedback']}")
        print(f"  Approved: {summary['approved']} ({summary['approval_rate']:.1f}%)")
        print(f"  Rejected: {summary['rejected']}")
        print(f"  Modified: {summary['modified']}")
        print(f"  Flagged: {summary['flagged']}")
        
        print(f"\n🔍 Identified Patterns: {len(patterns)}")
        for pattern in patterns:
            print(f"  • {pattern['pattern_type']}: {pattern.get('reason', pattern.get('modification_type'))}")
        
        print(f"\n💡 Improvement Recommendations:")
        for rec in recommendations:
            print(f"  {rec}")
        
        # Export to Excel
        print(f"\n📄 Exporting detailed report to {output_file}...")
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Sheet 1: Summary
            summary_df = pd.DataFrame([summary])
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Sheet 2: Feedback Details
            feedback_df = pd.DataFrame(self.feedback_collector.feedback_data['test_case_feedback'])
            if not feedback_df.empty:
                feedback_df.to_excel(writer, sheet_name='Feedback Details', index=False)
            
            # Sheet 3: Patterns
            if patterns:
                patterns_df = pd.DataFrame(patterns)
                patterns_df.to_excel(writer, sheet_name='Patterns', index=False)
            
            # Sheet 4: Recommendations
            rec_df = pd.DataFrame({'recommendation': recommendations})
            rec_df.to_excel(writer, sheet_name='Recommendations', index=False)
        
        print(f"✅ Report saved: {output_file}")


# ============================================
# MAIN EXECUTION
# ============================================

def main():
    """Main execution - Mock feedback loop demonstration"""
    
    print("\n" + "="*80)
    print(" " * 15 + "FEEDBACK LOOP DEMONSTRATION (MOCK)")
    print("="*80)
    print("\n⚠️  NOTE: This is a MOCK IMPLEMENTATION for academic demonstration.")
    print("   Production system would require real user feedback via UI/API.")
    print("="*80)
    
    # Load validation results from previous phase
    VALIDATION_FILE = "../06_Validation_QA/validation_report.xlsx"
    FEEDBACK_REPORT = "../07_Feedback_Loop/feedback_improvement_report.xlsx"
    
    Path("../07_Feedback_Loop").mkdir(parents=True, exist_ok=True)
    
    print("\n[1/3] Loading validation results...")
    validation_df = pd.read_excel(VALIDATION_FILE, sheet_name='Validation Results')
    validation_results = validation_df.to_dict('records')
    
    # Convert to expected format
    formatted_results = []
    for row in validation_results:
        formatted_results.append({
            'test_id': row['test_id'],
            'requirement_id': row['requirement_id'],
            'confidence': {
                'overall_score': row['overall_score'],
                'confidence_level': row['confidence_level']
            }
        })
    
    print(f"  Loaded {len(formatted_results)} validation results")
    
    # Initialize mock feedback loop
    print("\n[2/3] Simulating feedback collection...")
    feedback_loop = MockFeedbackLoop()
    feedback_loop.simulate_feedback_collection(formatted_results, n_samples=50)
    
    # Generate improvement report
    print("\n[3/3] Generating improvement analysis...")
    feedback_loop.generate_improvement_report(FEEDBACK_REPORT)
    
    print("\n" + "="*80)
    print(" " * 20 + "✅ FEEDBACK LOOP DEMONSTRATION COMPLETE")
    print("="*80)
    print("\n💡 FUTURE WORK:")
    print("  1. Implement real UI for QA engineers to provide feedback")
    print("  2. Integrate LoRA fine-tuning for model improvement")
    print("  3. Deploy as web service with feedback API")
    print("  4. Collect 100-500 feedback examples for ML training")
    print("  5. Implement Query-by-Committee ensemble methods")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
