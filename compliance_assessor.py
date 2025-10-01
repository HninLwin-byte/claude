from sentence_transformers import CrossEncoder
import torch
import json
from typing import List, Dict

def load_test_dataset(json_file_path: str) -> List[Dict]:
    """Load the test dataset from JSON file"""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data['test_dataset']['test_cases']
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return []

# def calculate_similarity_scores(test_cases: List[Dict]) -> List[Dict]:
#     """Calculate similarity scores between ATCO-Pilot pairs"""
    
   
#     print("Loading cross-encoder model...")
#     model = CrossEncoder("cross-encoder/ms-marco-electra-base", activation_fn=torch.nn.Sigmoid())
    
   
#     sentence_pairs = []
    
#     for case in test_cases:
#         if case['ATCO'] and case['ATCO_components']['altitude']:
#             atc_altitude = case['ATCO_components']['altitude']
#             pilot_altitude = case['PILOT_components']['altitude']
#             sentence_pairs.append((atc_altitude, pilot_altitude))
#             scores = model.predict(sentence_pairs)
#             results = []
#             for i, case in enumerate(test_cases):
#                 result = case.copy()
#                 result['similarity_score'] = float(scores[i])
#                 threshold = 0.5
#                 result['predicted_compliance'] = "Compliance" if scores[i] >= threshold else "Non-Compliance"
        
        
#                 result['correct_prediction'] = (
#                     result['predicted_compliance'] == result['Expected_compliance']
#                 )
                
#                 results.append(result)
                
                
               
            
#                 return results

#          if case['ATCO'] and case['ATCO_components']['heading_degree']:
#              atc_altitude = case['ATCO_components']['heading_degree']
#              pilot_altitude = case['PILOT_components']['heading_degree']
#              sentence_pairs.append((atc_altitude, pilot_altitude))
#              scores = model.predict(sentence_pairs)
#              results = []
#              for i, case in enumerate(test_cases):
#                 result = case.copy()
#                 result['similarity_score'] = float(scores[i])
#                 threshold = 0.5
#                 result['predicted_compliance'] = "Compliance" if scores[i] >= threshold else "Non-Compliance"
        
        
#                 result['correct_prediction'] = (
#                     result['predicted_compliance'] == result['Expected_compliance']
#                 )
                
#                 results.append(result)
                
                
               
            
#               return results

#           if case['ATCO'] and case['ATCO_components']['frequency']:
#              atc_altitude = case['ATCO_components']['frequency']
#              pilot_altitude = case['PILOT_components']['frequency']
#              sentence_pairs.append((atc_altitude, pilot_altitude))
#              scores = model.predict(sentence_pairs)
#              results = []
#              for i, case in enumerate(test_cases):
#                 result = case.copy()
#                 result['similarity_score'] = float(scores[i])
#                 threshold = 0.5
#                 result['predicted_compliance'] = "Compliance" if scores[i] >= threshold else "Non-Compliance"
        
        
#                 result['correct_prediction'] = (
#                     result['predicted_compliance'] == result['Expected_compliance']
#                 )
                
#                 results.append(result)
                
                
               
            
#               return results

def calculate_similarity_scores(test_cases: List[Dict]) -> List[Dict]:
      
      print("Loading cross-encoder model...")
      model = CrossEncoder("cross-encoder/ms-marco-electra-base", activation_fn=torch.nn.Sigmoid())

      results = []

      for case in test_cases:
          result = case.copy()
          component_scores = {}

          
          if case.get('ATCO') and case.get('ATCO_components', {}).get('altitude'):
              atc_altitude = case['ATCO_components']['altitude']
              pilot_altitude = case.get('PILOT_components', {}).get('altitude')

              if pilot_altitude:
                  score = model.predict([(atc_altitude, pilot_altitude)])[0]
                  component_scores['altitude_score'] = float(score)
              else:
                  component_scores['altitude_score'] = 0.0
          else:
              component_scores['altitude_score'] = None

         
          if case.get('ATCO') and case.get('ATCO_components', {}).get('heading_degree'):
              atc_heading = case['ATCO_components']['heading_degree']
              pilot_heading = case.get('PILOT_components', {}).get('heading_degree')

              if pilot_heading:
                  score = model.predict([(atc_heading, pilot_heading)])[0]
                  component_scores['heading_degree_score'] = float(score)
              else:
                  component_scores['heading_degree_score'] = 0.0
          else:
              component_scores['heading_degree_score'] = None

          
          if case.get('ATCO') and case.get('ATCO_components', {}).get('frequency'):
              atc_frequency = case['ATCO_components']['frequency']
              print("atc_freq",  atc_frequency)
              pilot_frequency = case.get('PILOT_components', {}).get('frequency')
              print("pilot_freq",  pilot_frequency)

              if pilot_frequency:
                  score = model.predict([(atc_frequency, pilot_frequency)])[0]
                  component_scores['frequency_score'] = float(score)
              else:
                  component_scores['frequency_score'] = 0.0
          else:
              component_scores['frequency_score'] = None

          
          result['component_scores'] = component_scores

          
          valid_scores = [score for score in component_scores.values() if score is not None]
          if valid_scores:
              overall_score = sum(valid_scores) / len(valid_scores)
          else:
              overall_score = 0.0

          result['similarity_score'] = overall_score

         
          threshold = 0.5
          result['predicted_compliance'] = "Compliance" if overall_score >= threshold else "Non-Compliance"

         
          result['correct_prediction'] = (
              result['predicted_compliance'] == result['Expected_compliance']
          )

          results.append(result)

      return results

            
                
                
                
            
            
            
            
            
    #     atco_instruction = case['ATCO']
    #     pilot_readback = case['PILOT']
    #     if 
    #     sentence_pairs.append((atco_instruction, pilot_readback))
    
    # print(f"Calculating similarity scores for {len(sentence_pairs)} pairs...")
    
    # # Get similarity scores in batches for efficiency
    # scores = model.predict(sentence_pairs)
    
    
    # results = []
    # for i, case in enumerate(test_cases):
    #     result = case.copy()
    #     result['similarity_score'] = float(scores[i])
        
        
    #     threshold = 0.99  
    #     result['predicted_compliance'] = "Compliance" if scores[i] >= threshold else "Non-Compliance"
        
        
    #     result['correct_prediction'] = (
    #         result['predicted_compliance'] == result['Expected_compliance']
    #     )
        
    #     results.append(result)
        
    #     # Print progress
    #     if (i + 1) % 10 == 0:
    #         print(f"Processed {i + 1}/{len(test_cases)} cases...")
    
    # return results

def analyze_results(results: List[Dict]) -> Dict:
    """Analyze the prediction results"""
    total_cases = len(results)
    correct_predictions = sum(1 for r in results if r['correct_prediction'])
    
    # Separate compliant and non-compliant cases
    compliant_cases = [r for r in results if r['Expected_compliance'] == 'Compliance']
    non_compliant_cases = [r for r in results if r['Expected_compliance'] == 'Non-Compliance']
    
    # Calculate statistics
    compliant_correct = sum(1 for r in compliant_cases if r['correct_prediction'])
    non_compliant_correct = sum(1 for r in non_compliant_cases if r['correct_prediction'])
    
    # Calculate average scores
    compliant_scores = [r['similarity_score'] for r in compliant_cases]
    non_compliant_scores = [r['similarity_score'] for r in non_compliant_cases]
    
    avg_compliant_score = sum(compliant_scores) / len(compliant_scores) if compliant_scores else 0
    avg_non_compliant_score = sum(non_compliant_scores) / len(non_compliant_scores) if non_compliant_scores else 0
    
    analysis = {
        'total_cases': total_cases,
        'correct_predictions': correct_predictions,
        'accuracy': correct_predictions / total_cases,
        'compliant_cases': {
            'total': len(compliant_cases),
            'correct': compliant_correct,
            'accuracy': compliant_correct / len(compliant_cases) if compliant_cases else 0,
            'avg_score': avg_compliant_score
        },
        'non_compliant_cases': {
            'total': len(non_compliant_cases),
            'correct': non_compliant_correct,
            'accuracy': non_compliant_correct / len(non_compliant_cases) if non_compliant_cases else 0,
            'avg_score': avg_non_compliant_score
        }
    }
    
    return analysis

def save_results(results: List[Dict], analysis: Dict, output_file: str):
    """Save results to JSON file"""
    output_data = {
        'analysis': analysis,
        'detailed_results': results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {output_file}")

def print_analysis(analysis: Dict):
    """Print analysis results"""
    print("\n" + "="*60)
    print("COMPLIANCE ASSESSMENT ANALYSIS")
    print("="*60)
    print(f"Total test cases: {analysis['total_cases']}")
    print(f"Correct predictions: {analysis['correct_predictions']}")
    print(f"Overall Accuracy: {analysis['accuracy']:.3f} ({analysis['accuracy']*100:.1f}%)")
    
    print(f"\nCompliant Cases:")
    print(f"  Total: {analysis['compliant_cases']['total']}")
    print(f"  Correct: {analysis['compliant_cases']['correct']}")
    print(f"  Accuracy: {analysis['compliant_cases']['accuracy']:.3f} ({analysis['compliant_cases']['accuracy']*100:.1f}%)")
    print(f"  Average Score: {analysis['compliant_cases']['avg_score']:.3f}")
    
    print(f"\nNon-Compliant Cases:")
    print(f"  Total: {analysis['non_compliant_cases']['total']}")
    print(f"  Correct: {analysis['non_compliant_cases']['correct']}")
    print(f"  Accuracy: {analysis['non_compliant_cases']['accuracy']:.3f} ({analysis['non_compliant_cases']['accuracy']*100:.1f}%)")
    print(f"  Average Score: {analysis['non_compliant_cases']['avg_score']:.3f}")

def main():
    """Main execution function"""

    input_file = "/workspace/Kris/WhisperATC/QWEN/test_data_compliance_assessor_entities.json"
    output_file = "/workspace/Kris/WhisperATC/QWEN/test_data_compliance_assessor_result.json"
    
    # Load test dataset
    test_cases = load_test_dataset(input_file)
    if not test_cases:
        print("No test cases loaded. Exiting.")
        return
    
    print(f"Loaded {len(test_cases)} test cases")
    
    # Calculate similarity scores
    results = calculate_similarity_scores(test_cases)
    
    # Analyze results
    analysis = analyze_results(results)
    
    # Print analysis
    print_analysis(analysis)
    
    # Save results
    save_results(results, analysis, output_file)
    
    # Print some example results
    print("\n" + "="*60)
    print("EXAMPLE RESULTS")
    print("="*60)
    for i in range(3):  # Show first 5 cases
        case = results[i]
        print(f"\nCase {case['id']}:")
        print(f"ATCO: {case['ATCO']}")
        print(f"PILOT: {case['PILOT']}")
        print(f"Similarity Score: {case['similarity_score']:.4f}")
        print(f"Expected: {case['Expected_compliance']}")
        print(f"Predicted: {case['predicted_compliance']}")
        print(f"Correct: {'✓' if case['correct_prediction'] else '✗'}")

if __name__ == "__main__":
    main()


    