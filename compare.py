import json
from collections import Counter
from typing import List, Dict, Any
import numpy as np
import scipy.stats as stats
import re
class HallucinationAnalyzer:
    def __init__(self, ground_truth_file: str, llm_output_files: List[str]):
        """
        Initialize the analyzer with ground truth and multiple LLM output JSON files
        
        :param ground_truth_file: Path to the ground truth JSON file
        :param llm_output_files: List of paths to the LLM output JSON files
        """
        with open(ground_truth_file, 'r') as f:
            self.ground_truth = f.read()
        
        self.ground_truth_data = json.loads(self.ground_truth)
        self.llm_outputs = []
        self.llm_output_names = []
        for llm_file in llm_output_files:
            with open(llm_file, 'r') as f:
                llm_data = json.loads(f.read())
                self.llm_outputs.append(llm_data)
                model_name = llm_file.split('/')[-1].split('.json')[0]
                print(model_name)
                self.llm_output_names.append(model_name)
        self.results = []
        for i in range(len(llm_output_files)):
            self.results.append({
                'model_name': self.llm_output_names[i],
                'total_images': 0,
                'hallucinated_images': 0,
                'perfect_matches': 0,
                'error_categories': {
                    'fabrication': 0,
                    'omission': 0
                },
                'normalized_hallucination_rates': [],
                'condition_breakdown': {
                    'dark': {'true': 0, 'false': 0},
                    'weather': {'rain': 0, 'clear': 0, 'fog': 0}
                },
                'image_hallucination_data': {}  
            })
        self.compare_results = []
    

    def parse_llm_objects(self, raw_output_value):
        """
        Parse LLM objects from the raw_output field using regex whether they are in a comma-separated string or list format
        Also handles empty values
        
        :param raw_output_value: The raw_output value from LLM output (string or list)
        :return: List of object strings
        """
        if raw_output_value == "":
            return []
        if not isinstance(raw_output_value, str):
            return ["malformed_format"]
        if isinstance(raw_output_value, str):
            return [obj.strip() for obj in re.split(r'\s*,\s*', raw_output_value) if obj.strip()]
            
        return []
    
    def parse_ground_truth_objects(self, objects_value):
        """
        Parse ground truth objects from array format or handle empty string
        
        :param objects_value: The objects value from ground truth
        :return: List of object strings
        """
        if objects_value is None or objects_value == "":
            return []
            
        if isinstance(objects_value, list):
            return objects_value
            
        return []
    
    def analyze_hallucinations(self):
        """
        Perform hallucination analysis across all images for all LLM outputs
        """
        ground_truth_dict = {item['id']: item for item in self.ground_truth_data}
        for llm_index, llm_output_data in enumerate(self.llm_outputs):
            llm_output_dict = {item['id']: item for item in llm_output_data}
            model_compare_results = []
            for image_id, ground_truth_entry in ground_truth_dict.items():
                if image_id not in llm_output_dict:
                    continue
                self.results[llm_index]['total_images'] += 1
                ground_truth_objects = self.parse_ground_truth_objects(ground_truth_entry.get('objects', []))
                llm_objects = self.parse_llm_objects(llm_output_dict[image_id].get('raw_output', ""))
                gt_counter = Counter(ground_truth_objects)
                llm_counter = Counter(llm_objects)
                fabricated_objects = list((llm_counter - gt_counter).elements())
                omitted_objects = list((gt_counter - llm_counter).elements())
                
                model_name=self.llm_output_names[llm_index]
                if model_name=="model_outputs_gpt-4":
                    if len(fabricated_objects)>0:
                        print("---")
                        print(image_id)
                        print(llm_objects)                    
                total_errors = len(fabricated_objects) + len(omitted_objects)
                is_dark = str(ground_truth_entry.get('dark', False)).lower()  # 'true' or 'false'
                weather = ground_truth_entry.get('weather', 'unknown')
                if total_errors > 0:
                    if is_dark in ['true', 'false']:
                        self.results[llm_index]['condition_breakdown']['dark'][is_dark] += 1
                    if weather in ['rain', 'clear', 'fog']:
                        self.results[llm_index]['condition_breakdown']['weather'][weather] += 1
                compare_result = {
                    'id': image_id,
                    'model': self.llm_output_names[llm_index],
                    'perfect_match': total_errors == 0,
                    'error_categories': {
                        'fabrication': len(fabricated_objects),
                        'omission': len(omitted_objects)
                    }
                }
                model_compare_results.append(compare_result)
                if total_errors == 0:
                    self.results[llm_index]['perfect_matches'] += 1
                else:
                    self.results[llm_index]['hallucinated_images'] += 1
                    self.results[llm_index]['error_categories']['fabrication'] += len(fabricated_objects)
                    self.results[llm_index]['error_categories']['omission'] += len(omitted_objects)
                union_counter = gt_counter | llm_counter  # This takes maximum count of each element
                union_size = sum(union_counter.values())  # Total count of all objects in union
                normalized_hallucination_rate = total_errors / union_size if union_size > 0 else 0
                self.results[llm_index]['normalized_hallucination_rates'].append(normalized_hallucination_rate)
                self.results[llm_index]['image_hallucination_data'][image_id] = {
                    'normalized_rate': normalized_hallucination_rate,
                    'total_errors': total_errors,
                    'fabrication': len(fabricated_objects),
                    'omission': len(omitted_objects)
                }
            self.compare_results.extend(model_compare_results)
    
    def compute_aggregate_metrics(self):
        """
        Compute aggregate metrics for each LLM dataset
        """
        for llm_index in range(len(self.llm_outputs)):
            total_images = self.results[llm_index]['total_images']
            if self.results[llm_index]['normalized_hallucination_rates']:
                self.results[llm_index]['average_normalized_hallucination_rate'] = sum(
                    self.results[llm_index]['normalized_hallucination_rates']
                ) / len(self.results[llm_index]['normalized_hallucination_rates'])
            else:
                self.results[llm_index]['average_normalized_hallucination_rate'] = 0
            
            self.results[llm_index]['hallucination_rate'] = (
                self.results[llm_index]['hallucinated_images'] / total_images
            ) if total_images > 0 else 0
            
            self.results[llm_index]['perfect_match_rate'] = (
                self.results[llm_index]['perfect_matches'] / total_images
            ) if total_images > 0 else 0
            
    def save_compare_results(self):
        """
        Save compare_results to a JSON file
        """
        with open('compare_results.json', 'w') as f:
            json.dump(self.compare_results, f, indent=2)
        print(f"\nComparison results saved to compare_results.json")
    
    def display_results(self):
        """
        Display detailed analysis results for each LLM
        """
        for llm_index in range(len(self.llm_outputs)):
            model_name = self.results[llm_index]['model_name']
            print(f"\nHallucination Analysis Results for {model_name}:")
            print("-" * 60)
            print("Standard Metrics (Exact Match Required):")
            print(f"Total Images Analyzed: {self.results[llm_index]['total_images']}")
            print(f"Perfect Matches: {self.results[llm_index]['perfect_matches']} ({self.results[llm_index]['perfect_match_rate']:.2%})")
            print(f"Hallucinated Images: {self.results[llm_index]['hallucinated_images']} ({self.results[llm_index]['hallucination_rate']:.2%})")
            print(f"Average Normalized Hallucination Rate: {self.results[llm_index]['average_normalized_hallucination_rate']:.4f}")
            
            print("\nError Categories:")
            for category, count in self.results[llm_index]['error_categories'].items():
                print(f"- {category.capitalize()}: {count}")

            print("\nHallucination Breakdown by Light and Weather Conditions:")
            print("By Light Condition:")
            for light_cond, count in self.results[llm_index]['condition_breakdown']['dark'].items():
                print(f"- Dark = {light_cond.capitalize()}: {count} hallucinated images")

            print("By Weather Condition:")
            for weather_cond, count in self.results[llm_index]['condition_breakdown']['weather'].items():
                print(f"- {weather_cond.capitalize()}: {count} hallucinated images")
        if len(self.llm_outputs) > 1:
            self.display_model_comparison()
    
    def display_model_comparison(self):
        """
        Display a comparison of metrics between the LLM models
        """
        print("\n" + "=" * 80)
        print("MODEL COMPARISON SUMMARY")
        print("=" * 80)
        metrics = [
            ('Perfect Match Rate', 'perfect_match_rate', '{:.2%}'),
            ('Hallucination Rate', 'hallucination_rate', '{:.2%}'),
            ('Avg. Normalized Hallucination Rate', 'average_normalized_hallucination_rate', '{:.4f}'),
        ]
        model_col_width = max(len(model['model_name']) for model in self.results) + 2
        metric_col_width = max(len(m[0]) for m in metrics) + 2
        value_col_width = 10
        print(f"{'Metric':<{metric_col_width}}", end="")
        for model in self.results:
            print(f"{model['model_name']:<{model_col_width}}", end="")
        print()
        print("-" * (metric_col_width + model_col_width * len(self.results)))
        for metric_name, metric_key, format_str in metrics:
            print(f"{metric_name:<{metric_col_width}}", end="")
            for model in self.results:
                value = model.get(metric_key, 0)
                formatted_value = format_str.format(value)
                print(f"{formatted_value:<{model_col_width}}", end="")
            print()
        print("\nError Categories:")
        for category in ['fabrication', 'omission']:
            print(f"{category.capitalize():<{metric_col_width}}", end="")
            for model in self.results:
                value = model['error_categories'].get(category, 0)
                print(f"{value:<{model_col_width}}", end="")
            print()
    
    def perform_statistical_tests(self):
        """
        Perform statistical t-tests to compare hallucination rates between LLM models
        """
        if len(self.llm_outputs) < 2:
            print("Statistical tests require at least 2 models to compare")
            return
            
        print("\n" + "=" * 80)
        print("STATISTICAL SIGNIFICANCE TESTING")
        print("=" * 80)
        model_names = [result['model_name'] for result in self.results]
        print(f"{'Comparison':<30} {'t-statistic':<15} {'p-value':<15} {'Significant?':<10}")
        print("-" * 70)
        num_comparisons = (len(model_names) * (len(model_names) - 1)) // 2
        alpha_corrected = 0.05 / num_comparisons
        print("\nPaired t-tests (same images across models):")
        self._perform_paired_ttests(model_names, alpha_corrected)
        
        print("\nIndependent t-tests (for comparison):")
        self._perform_independent_ttests(model_names, alpha_corrected)
        
    def _perform_paired_ttests(self, model_names, alpha_corrected):
        """
        Perform paired t-tests between models using only images that exist in both datasets
        """
        for i in range(len(model_names)):
            for j in range(i+1, len(model_names)):
                image_ids_i = set(self.results[i]['image_hallucination_data'].keys())
                image_ids_j = set(self.results[j]['image_hallucination_data'].keys())
                common_image_ids = image_ids_i.intersection(image_ids_j)
                
                if len(common_image_ids) > 1:  # Need at least 2 samples for t-test
                    rates_i = [self.results[i]['image_hallucination_data'][img_id]['normalized_rate'] for img_id in common_image_ids]
                    rates_j = [self.results[j]['image_hallucination_data'][img_id]['normalized_rate'] for img_id in common_image_ids]
                    t_stat, p_value = stats.ttest_rel(rates_i, rates_j)
                    is_significant = p_value < alpha_corrected
                    
                    print(f"{model_names[i]} vs {model_names[j]:<15} {t_stat:<15.4f} {p_value:<15.4e} {'Yes' if is_significant else 'No'}")
                    print(f"  Mean difference: {np.mean(rates_i) - np.mean(rates_j):.4f}")
                    print(f"  Effect size (Cohen's d): {self._cohens_d(rates_i, rates_j):.4f}")
                    print(f"  Sample size: {len(common_image_ids)}")
                else:
                    print(f"{model_names[i]} vs {model_names[j]:<15} {'Insufficient common data':<30}")
    
    def _perform_independent_ttests(self, model_names, alpha_corrected):
        """
        Perform independent t-tests between models (less statistically powerful than paired)
        """
        for i in range(len(model_names)):
            for j in range(i+1, len(model_names)):
                rates_i = self.results[i]['normalized_hallucination_rates']
                rates_j = self.results[j]['normalized_hallucination_rates']
                
                if len(rates_i) > 1 and len(rates_j) > 1:  # Need at least 2 samples for t-test
                    t_stat, p_value = stats.ttest_ind(rates_i, rates_j, equal_var=False)
                    is_significant = p_value < alpha_corrected
                    
                    print(f"{model_names[i]} vs {model_names[j]:<15} {t_stat:<15.4f} {p_value:<15.4e} {'Yes' if is_significant else 'No'}")
                else:
                    print(f"{model_names[i]} vs {model_names[j]:<15} {'Insufficient data':<30}")
    
    def _cohens_d(self, x, y):
        """
        Calculate Cohen's d effect size
        """
        nx = len(x)
        ny = len(y)
        dof = nx + ny - 2
        pooled_std = np.sqrt(((nx-1)*np.var(x, ddof=1) + (ny-1)*np.var(y, ddof=1)) / dof)
        return (np.mean(x) - np.mean(y)) / pooled_std if pooled_std > 0 else 0
    
    def perform_correlation_analysis(self):
        """
        Analyze correlation between environmental factors and hallucination rates
        """
        print("\n" + "=" * 80)
        print("ENVIRONMENTAL FACTOR CORRELATION ANALYSIS")
        print("=" * 80)
        for llm_index, model_data in enumerate(self.results):
            model_name = model_data['model_name']
            print(f"\nAnalysis for {model_name}:")
            print("-" * 40)
            image_data = model_data['image_hallucination_data']
            dark_true_rates = []
            dark_false_rates = []
            
            weather_clear_rates = []
            weather_rain_rates = []
            weather_fog_rates = []
            for img_id, hall_data in image_data.items():
                for gt_item in self.ground_truth_data:
                    if gt_item['id'] == img_id:
                        is_dark = str(gt_item.get('dark', False)).lower() == 'true'
                        weather = gt_item.get('weather', 'unknown')
                        if is_dark:
                            dark_true_rates.append(hall_data['normalized_rate'])
                        else:
                            dark_false_rates.append(hall_data['normalized_rate'])
                            
                        if weather == 'clear':
                            weather_clear_rates.append(hall_data['normalized_rate'])
                        elif weather == 'rain':
                            weather_rain_rates.append(hall_data['normalized_rate'])
                        elif weather == 'fog':
                            weather_fog_rates.append(hall_data['normalized_rate'])
                        
                        break
            if len(dark_true_rates) > 1 and len(dark_false_rates) > 1:
                t_stat, p_value = stats.ttest_ind(dark_true_rates, dark_false_rates, equal_var=False)
                print(f"Dark vs Light Conditions: t={t_stat:.4f}, p={p_value:.4e}")
                print(f"  Dark images (n={len(dark_true_rates)}): Mean rate = {np.mean(dark_true_rates):.4f}")
                print(f"  Light images (n={len(dark_false_rates)}): Mean rate = {np.mean(dark_false_rates):.4f}")
                print(f"  Significant difference: {'Yes' if p_value < 0.05 else 'No'}")
            else:
                print("Insufficient data to compare dark vs light conditions")
            weather_data = [
                ("Clear", weather_clear_rates),
                ("Rain", weather_rain_rates),
                ("Fog", weather_fog_rates)
            ]
            valid_weather = [(name, rates) for name, rates in weather_data if len(rates) > 1]
            
            if len(valid_weather) > 1:
                print("\nWeather Condition Comparison:")
                for name, rates in valid_weather:
                    print(f"  {name} (n={len(rates)}): Mean rate = {np.mean(rates):.4f}")
                if len(valid_weather) == 2:
                    name1, rates1 = valid_weather[0]
                    name2, rates2 = valid_weather[1]
                    t_stat, p_value = stats.ttest_ind(rates1, rates2, equal_var=False)
                    print(f"  {name1} vs {name2}: t={t_stat:.4f}, p={p_value:.4e}")
                    print(f"  Significant difference: {'Yes' if p_value < 0.05 else 'No'}")
            else:
                print("\nInsufficient data to compare weather conditions")
    
    def run(self):
        """
        Execute full hallucination analysis
        """
        self.analyze_hallucinations()
        self.compute_aggregate_metrics()
        self.display_results()
        self.perform_statistical_tests()
        self.perform_correlation_analysis()
        self.save_compare_results()
        self.perform_friedman_test()
        self.test_normality()
    def test_normality(self):
        """Test if hallucination rates follow normal distribution"""
        from scipy import stats
        import matplotlib.pyplot as plt
        
        print("\n" + "=" * 80)
        print("NORMALITY TESTING")
        print("=" * 80)
        
        for llm_index, model_data in enumerate(self.results):
            model_name = model_data['model_name']
            rates = model_data['normalized_hallucination_rates']
            shapiro_stat, shapiro_p = stats.shapiro(rates)
            
            print(f"\nNormality tests for {model_name}:")
            print(f"Shapiro-Wilk test: W={shapiro_stat:.4f}, p={shapiro_p:.4e}")
            print(f"Data normally distributed: {'Yes' if shapiro_p >= 0.05 else 'No'}")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            ax1.hist(rates, bins=30, alpha=0.7, density=True)
            ax1.set_title(f"Distribution of hallucination rates - {model_name}")
            ax1.set_xlabel("Normalized hallucination rate")
            ax1.set_ylabel("Frequency")
            stats.probplot(rates, plot=ax2)
            ax2.set_title(f"Q-Q Plot - {model_name}")
            
            plt.tight_layout()
            plt.savefig(f"normality_test_{model_name}.png")
            plt.close()
    def perform_friedman_test(self):
        """
        Perform Friedman test to compare hallucination rates across all models simultaneously
        """
        import scipy.stats as stats
        import numpy as np
        import pandas as pd
        
        print("\n" + "=" * 80)
        print("FRIEDMAN TEST FOR MODEL COMPARISON")
        print("=" * 80)
        all_image_ids = []
        for result in self.results:
            all_image_ids.append(set(result['image_hallucination_data'].keys()))
        

        n_images_first=0
        for i in range(len(all_image_ids)):
            if n_images_first>0:
                if n_images_first!=len(all_image_ids[i]):
                    print("Mismatch in set!")
                    return
            n_images_first=len(all_image_ids[i])

        

        common_image_ids = set.intersection(*all_image_ids)

        if len(common_image_ids) < 5:
            print("Insufficient common data points across all models for Friedman test")
            return
        common_image_ids = sorted(list(common_image_ids))
        model_names = [result['model_name'] for result in self.results]
        data_matrix = np.zeros((len(common_image_ids), len(model_names)))
        
        for i, image_id in enumerate(common_image_ids):
            for j, result in enumerate(self.results):
                data_matrix[i, j] = result['image_hallucination_data'][image_id]['normalized_rate']
        friedman_stat, friedman_p = stats.friedmanchisquare(*[data_matrix[:, j] for j in range(data_matrix.shape[1])])
        n = len(common_image_ids)  # number of blocks/subjects
        k = len(model_names)       # number of conditions/treatments
        ranks = np.zeros_like(data_matrix)
        for i in range(n):
            ranks[i, :] = stats.rankdata(data_matrix[i, :])
        kendall_w = friedman_stat / (n * (k-1))
        print(f"Number of images analyzed: {len(common_image_ids)}")
        print(f"Friedman chi-square statistic: {friedman_stat:.4f}")
        print(f"p-value: {friedman_p:.4e}")
        print(f"Significant difference across models: {'Yes' if friedman_p < 0.05 else 'No'}")
        print(f"Kendall's W (effect size): {kendall_w:.4f} - {'Strong' if kendall_w > 0.5 else 'Moderate' if kendall_w > 0.3 else 'Weak'} effect")
        print("\nDescriptive Statistics:")
        for j, model in enumerate(model_names):
            mean_rate = np.mean(data_matrix[:, j])
            median_rate = np.median(data_matrix[:, j])
            mean_rank = np.mean(ranks[:, j])
            print(f"{model}: Mean={mean_rate:.4f}, Median={median_rate:.4f}, Mean Rank={mean_rank:.2f}")
        if friedman_p < 0.05:
            print("\nPost-hoc Analysis (Conover test):")
            self._perform_conover_posthoc(ranks, n, k, model_names)
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 6))
            df = pd.DataFrame(data_matrix, columns=model_names)
            df.boxplot()
            plt.title('Hallucination Rates Across Models')
            plt.ylabel('Normalized Hallucination Rate')
            plt.grid(False)
            plt.savefig('friedman_boxplot.png')
            plt.close()
            
            print("\nBoxplot visualization saved as 'friedman_boxplot.png'")
        except ImportError:
            print("\nMatplotlib not available for visualization")

    def _perform_conover_posthoc(self, ranks, n, k, model_names):
        """
        Perform Conover post-hoc test for pairwise comparisons after a significant Friedman test
        
        :param ranks: Matrix of ranks (images × models)
        :param n: Number of blocks (images)
        :param k: Number of conditions (models)
        :param model_names: List of model names
        """
        import numpy as np
        import scipy.stats as stats
        mean_ranks = np.mean(ranks, axis=0)
        SSr = np.sum(ranks**2)
        A = n * k * (k + 1) * (k - 1) / 12
        B = SSr - n * k * (k + 1)**2 / 4
        C = B / ((k - 1) * (n - 1))
        num_comparisons = (k * (k - 1)) // 2
        alpha_corrected = 0.05 / num_comparisons
        t_critical = stats.t.ppf(1 - alpha_corrected/2, (n-1)*(k-1))
        print(f"{'Comparison':<30} {'Mean Rank Diff':<15} {'t-statistic':<15} {'p-value':<15} {'Significant?':<10}")
        print("-" * 85)
        for i in range(k):
            for j in range(i+1, k):
                mean_rank_diff = mean_ranks[i] - mean_ranks[j]
                t_stat = mean_rank_diff / np.sqrt(C * 2 / n)
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), (n-1)*(k-1)))
                is_significant = p_value < alpha_corrected
                
                print(f"{model_names[i]} vs {model_names[j]:<15} {mean_rank_diff:<15.4f} {t_stat:<15.4f} {p_value:<15.4e} {'Yes' if is_significant else 'No'}")
def main():
    import sys
    if len(sys.argv) >= 4:
        ground_truth_file = sys.argv[1]
        llm_output_files = sys.argv[2:]
    else:
        ground_truth_file = 'labels_processed.json'
        llm_output_files = ['model_outputs_gpt-4o.json', 'model_outputs_llava.json','model_outputs_gpt-4.5-preview.json']
        print(f"Using default file paths. To specify different files, run:")
        print(f"python {sys.argv[0]} <ground_truth_file> <llm_output_file1> <llm_output_file2> [<llm_output_file3> ...]")
    analyzer = HallucinationAnalyzer(ground_truth_file, llm_output_files)
    analyzer.run()

if __name__ == "__main__":
    main()