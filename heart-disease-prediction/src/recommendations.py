class HeartRecommendationSystem:
    """
    Professional Heart Disease Recommendation System
    Provides evidence-based recommendations for heart disease risk factors
    """
    
    def __init__(self):
        self.recommendations = {
            'Age': {
                'general': "Age is a major non-modifiable risk factor. Risk increases after 45 (men) and 55 (women).",
                'categories': {
                    '<40': "Focus on prevention and healthy lifestyle",
                    '40-54': "Regular checkups and risk assessment",
                    '55-64': "More frequent monitoring recommended",
                    '≥65': "Comprehensive cardiovascular evaluation needed"
                }
            },
            'Cholesterol': {
                'general': "High cholesterol directly contributes to atherosclerosis.",
                'categories': {
                    '<200': "Normal range",
                    '200-239': "Borderline high - lifestyle changes recommended",
                    '≥240': "High - consider medication along with lifestyle changes"
                }
            },
            'ChestPainType': {
                'general': "Type of chest pain is a key diagnostic factor.",
                'categories': {
                    '0': "Typical angina - high suspicion for CAD",
                    '1': "Atypical angina - intermediate probability",
                    '2': "Non-anginal pain - low probability",
                    '3': "Asymptomatic - needs risk factor evaluation"
                }
            },
            'ExerciseAngina': {
                'general': "Exercise-induced angina suggests significant coronary stenosis.",
                'categories': {
                    '0': "No exercise-induced angina",
                    '1': "Exercise-induced angina present - urgent evaluation needed"
                }
            },
            'ST_Slope': {
                'general': "ST segment slope during exercise indicates ischemia.",
                'categories': {
                    '0': "Upsloping - normal",
                    '1': "Flat - possible ischemia",
                    '2': "Downsloping - strong evidence of ischemia"
                }
            }
        }
    
    def get_recommendation(self, feature, value):
        """Get recommendation for a specific feature and value"""
        if feature not in self.recommendations:
            return f"No recommendations available for {feature}"
        
        rec = self.recommendations[feature]
        result = [f"Recommendations for {feature} (value: {value}):"]
        result.append(f"General: {rec['general']}")
        
        # Handle different feature types
        if feature == 'Age':
            if value < 40:
                category = '<40'
            elif 40 <= value < 55:
                category = '40-54'
            elif 55 <= value < 65:
                category = '55-64'
            else:
                category = '≥65'
            result.append(f"Age {value}: {rec['categories'][category]}")
        
        elif feature == 'Cholesterol':
            if value < 200:
                category = '<200'
            elif 200 <= value < 240:
                category = '200-239'
            else:
                category = '≥240'
            result.append(f"Cholesterol {value}: {rec['categories'][category]}")
        
        else:  # Categorical features
            value_str = str(value)
            if value_str in rec['categories']:
                result.append(f"Finding: {rec['categories'][value_str]}")
            else:
                result.append(f"Unknown value {value} for {feature}")
        
        return '\n'.join(result)
    
    def get_comprehensive_recommendation(self, patient_data):
        """Get recommendations for all features in patient data"""
        results = ["Comprehensive Heart Health Recommendations:", "="*50]
        
        for feature, value in patient_data.items():
            if feature in self.recommendations:
                results.append("")
                results.append(self.get_recommendation(feature, value))
        
        results.append("\nGeneral Advice:")
        results.append("- Regular exercise (150 mins/week moderate intensity)")
        results.append("- Heart-healthy diet (Mediterranean diet recommended)")
        results.append("- Smoking cessation if applicable")
        results.append("- Stress management and quality sleep")
        results.append("- Annual checkups with your physician")
        
        return '\n'.join(results)