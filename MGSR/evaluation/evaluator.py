import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import time
import json
import os
from collections import defaultdict
import logging

try:
    from plotnine import *
    PLOTNINE_AVAILABLE = True
except ImportError:
    PLOTNINE_AVAILABLE = False
try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

class MGSREvaluator:
    def __init__(self, 
                 model: nn.Module,
                 device: torch.device,
                 output_dir: str = './evaluation_results'):
        
        self.model = model
        self.device = device
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.logger = self._setup_logger()
        
        self.evaluation_results = {}
        self.baseline_results = {}
        
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger('MGSR_Evaluator')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def comprehensive_evaluation(self,
                                test_dataloader,
                                baseline_models: Dict[str, nn.Module] = None) -> Dict[str, Any]:
        
        self.logger.info("Starting comprehensive evaluation...")
        
        mgsr_results = self.evaluate_model(self.model, test_dataloader, "MGSR")
        
        if baseline_models:
            for name, baseline_model in baseline_models.items():
                baseline_results = self.evaluate_model(baseline_model, test_dataloader, name)
                self.baseline_results[name] = baseline_results
        
        self.evaluation_results['MGSR'] = mgsr_results
        
        comparison_results = self.compare_models()
        
        self.generate_evaluation_report()
        
        return {
            'mgsr_results': mgsr_results,
            'baseline_results': self.baseline_results,
            'comparison_results': comparison_results
        }
    
    def evaluate_model(self,
                      model: nn.Module,
                      test_dataloader,
                      model_name: str) -> Dict[str, Any]:
        
        self.logger.info(f"Evaluating {model_name}...")
        
        model.eval()
        
        all_predictions = []
        all_targets = []
        all_user_embeddings = []
        all_item_embeddings = []
        
        inference_times = []
        memory_usage = []
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(test_dataloader):
                batch_data = self._move_batch_to_device(batch_data)
                
                start_time = time.time()
                
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                
                outputs = model(**batch_data, mode='inference')
                
                end_time = time.time()
                inference_times.append(end_time - start_time)
                
                if torch.cuda.is_available():
                    memory_usage.append(torch.cuda.max_memory_allocated())
                
                if 'final_rankings' in outputs:
                    all_predictions.extend(outputs['final_rankings'])
                
                if 'target_items' in batch_data:
                    all_targets.extend(batch_data['target_items'].cpu().tolist())
                
                if 'user_embeddings' in outputs:
                    all_user_embeddings.append(outputs['user_embeddings'].cpu())
                
                if 'item_embeddings' in outputs:
                    all_item_embeddings.append(outputs['item_embeddings'].cpu())
        
        ranking_metrics = self.compute_ranking_metrics(all_predictions, all_targets)
        
        diversity_metrics = self.compute_diversity_metrics(all_predictions)
        
        novelty_metrics = self.compute_novelty_metrics(all_predictions, all_targets)
        
        coverage_metrics = self.compute_coverage_metrics(all_predictions)
        
        performance_metrics = {
            'avg_inference_time_ms': np.mean(inference_times) * 1000,
            'std_inference_time_ms': np.std(inference_times) * 1000,
            'avg_memory_usage_mb': np.mean(memory_usage) / (1024 * 1024) if memory_usage else 0,
            'throughput_qps': len(all_predictions) / sum(inference_times) if inference_times else 0
        }
        
        results = {
            'ranking_metrics': ranking_metrics,
            'diversity_metrics': diversity_metrics,
            'novelty_metrics': novelty_metrics,
            'coverage_metrics': coverage_metrics,
            'performance_metrics': performance_metrics,
            'predictions': all_predictions[:100],
            'targets': all_targets[:100]
        }
        
        if all_user_embeddings:
            embedding_analysis = self.analyze_embeddings(
                torch.cat(all_user_embeddings, dim=0),
                torch.cat(all_item_embeddings, dim=0) if all_item_embeddings else None
            )
            results['embedding_analysis'] = embedding_analysis
        
        return results
    
    def compute_ranking_metrics(self,
                               predictions: List[List[int]],
                               targets: List[int]) -> Dict[str, float]:
        
        if not predictions or not targets:
            return {f'{metric}@{k}': 0.0 for metric in ['HR', 'NDCG', 'Precision', 'Recall'] 
                   for k in [1, 5, 10, 20, 50]}
        
        metrics = {}
        k_values = [1, 5, 10, 20, 50]
        
        for k in k_values:
            hr_scores = []
            ndcg_scores = []
            precision_scores = []
            recall_scores = []
            
            for pred, target in zip(predictions, targets):
                if not pred:
                    hr_scores.append(0.0)
                    ndcg_scores.append(0.0)
                    precision_scores.append(0.0)
                    recall_scores.append(0.0)
                    continue
                
                pred_k = pred[:k]
                
                hr = 1.0 if target in pred_k else 0.0
                hr_scores.append(hr)
                
                ndcg = self._compute_ndcg(pred_k, [target], k)
                ndcg_scores.append(ndcg)
                
                precision = 1.0 / k if target in pred_k else 0.0
                precision_scores.append(precision)
                
                recall = 1.0 if target in pred_k else 0.0
                recall_scores.append(recall)
            
            metrics[f'HR@{k}'] = np.mean(hr_scores)
            metrics[f'NDCG@{k}'] = np.mean(ndcg_scores)
            metrics[f'Precision@{k}'] = np.mean(precision_scores)
            metrics[f'Recall@{k}'] = np.mean(recall_scores)
        
        mrr_scores = []
        for pred, target in zip(predictions, targets):
            if target in pred:
                rank = pred.index(target) + 1
                mrr_scores.append(1.0 / rank)
            else:
                mrr_scores.append(0.0)
        
        metrics['MRR'] = np.mean(mrr_scores)
        
        return metrics
    
    def compute_diversity_metrics(self, predictions: List[List[int]]) -> Dict[str, float]:
        
        if not predictions:
            return {'intra_list_diversity': 0.0, 'inter_list_diversity': 0.0}
        
        intra_diversities = []
        all_items = set()
        
        for pred in predictions:
            if len(pred) <= 1:
                intra_diversities.append(0.0)
                continue
            
            unique_items = len(set(pred))
            total_items = len(pred)
            intra_diversity = unique_items / total_items
            intra_diversities.append(intra_diversity)
            
            all_items.update(pred)
        
        avg_intra_diversity = np.mean(intra_diversities)
        
        total_recommendations = sum(len(pred) for pred in predictions)
        inter_diversity = len(all_items) / total_recommendations if total_recommendations > 0 else 0.0
        
        return {
            'intra_list_diversity': avg_intra_diversity,
            'inter_list_diversity': inter_diversity,
            'catalog_coverage': len(all_items)
        }
    
    def compute_novelty_metrics(self,
                               predictions: List[List[int]],
                               targets: List[int]) -> Dict[str, float]:
        
        if not predictions or not targets:
            return {'novelty': 0.0, 'serendipity': 0.0}
        
        item_popularity = defaultdict(int)
        for target in targets:
            item_popularity[target] += 1
        
        total_users = len(targets)
        
        novelty_scores = []
        serendipity_scores = []
        
        for pred, target in zip(predictions, targets):
            if not pred:
                novelty_scores.append(0.0)
                serendipity_scores.append(0.0)
                continue
            
            pred_novelty = []
            for item in pred[:10]:
                popularity = item_popularity.get(item, 0)
                if popularity > 0:
                    novelty = -np.log2(popularity / total_users)
                else:
                    novelty = np.log2(total_users)
                pred_novelty.append(novelty)
            
            avg_novelty = np.mean(pred_novelty) if pred_novelty else 0.0
            novelty_scores.append(avg_novelty)
            
            unexpected_items = [item for item in pred[:10] if item != target]
            serendipity = len(unexpected_items) / min(10, len(pred))
            serendipity_scores.append(serendipity)
        
        return {
            'novelty': np.mean(novelty_scores),
            'serendipity': np.mean(serendipity_scores)
        }
    
    def compute_coverage_metrics(self, predictions: List[List[int]]) -> Dict[str, float]:
        
        if not predictions:
            return {'catalog_coverage': 0.0, 'gini_coefficient': 1.0}
        
        all_recommended_items = []
        for pred in predictions:
            all_recommended_items.extend(pred[:10])
        
        unique_items = set(all_recommended_items)
        catalog_coverage = len(unique_items)
        
        item_counts = defaultdict(int)
        for item in all_recommended_items:
            item_counts[item] += 1
        
        if len(item_counts) <= 1:
            gini_coefficient = 0.0
        else:
            counts = sorted(item_counts.values())
            n = len(counts)
            cumsum = np.cumsum(counts)
            gini_coefficient = (2 * np.sum((np.arange(1, n + 1) * counts))) / (n * cumsum[-1]) - (n + 1) / n
        
        return {
            'catalog_coverage': catalog_coverage,
            'gini_coefficient': gini_coefficient
        }
    
    def analyze_embeddings(self,
                          user_embeddings: torch.Tensor,
                          item_embeddings: torch.Tensor = None) -> Dict[str, Any]:
        
        user_emb_np = user_embeddings.numpy()
        if not SKLEARN_AVAILABLE:
            self.logger.warning("scikit-learn is unavailable; PCA and t-SNE were skipped")
            return {
                'user_embedding_variance': np.var(user_emb_np, axis=0).mean(),
                'user_embedding_mean_norm': np.linalg.norm(user_emb_np, axis=1).mean()
            }
        user_sample = user_emb_np[:1000]
        if len(user_sample) < 2:
            return {
                'user_embedding_variance': np.var(user_emb_np, axis=0).mean(),
                'user_embedding_mean_norm': np.linalg.norm(user_emb_np, axis=1).mean()
            }
        
        pca = PCA(n_components=2)
        user_pca = pca.fit_transform(user_sample)
        
        tsne = TSNE(
            n_components=2,
            random_state=42,
            perplexity=min(30, len(user_sample) - 1)
        )
        user_tsne = tsne.fit_transform(user_sample)
        
        analysis = {
            'user_embedding_variance': np.var(user_emb_np, axis=0).mean(),
            'user_embedding_mean_norm': np.linalg.norm(user_emb_np, axis=1).mean(),
            'pca_explained_variance': pca.explained_variance_ratio_.tolist(),
            'user_pca_coordinates': user_pca.tolist(),
            'user_tsne_coordinates': user_tsne.tolist()
        }
        
        if item_embeddings is not None:
            item_emb_np = item_embeddings.numpy()
            
            item_sample = item_emb_np[:1000]
            item_pca = pca.transform(item_sample)
            item_tsne = TSNE(
                n_components=2,
                random_state=42,
                perplexity=min(30, len(item_sample) - 1)
            ).fit_transform(item_sample) if len(item_sample) > 1 else np.zeros((len(item_sample), 2))
            
            analysis.update({
                'item_embedding_variance': np.var(item_emb_np, axis=0).mean(),
                'item_embedding_mean_norm': np.linalg.norm(item_emb_np, axis=1).mean(),
                'item_pca_coordinates': item_pca.tolist(),
                'item_tsne_coordinates': item_tsne.tolist()
            })
        
        return analysis
    
    def compare_models(self) -> Dict[str, Any]:
        
        if not self.baseline_results:
            return {}
        
        comparison_data = []
        
        for model_name, results in {**self.baseline_results, 'MGSR': self.evaluation_results['MGSR']}.items():
            ranking_metrics = results['ranking_metrics']
            performance_metrics = results['performance_metrics']
            
            comparison_data.append({
                'Model': model_name,
                'HR@10': ranking_metrics.get('HR@10', 0.0),
                'NDCG@10': ranking_metrics.get('NDCG@10', 0.0),
                'MRR': ranking_metrics.get('MRR', 0.0),
                'Inference_Time_ms': performance_metrics.get('avg_inference_time_ms', 0.0),
                'Memory_Usage_MB': performance_metrics.get('avg_memory_usage_mb', 0.0),
                'Throughput_QPS': performance_metrics.get('throughput_qps', 0.0)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        self.generate_comparison_plots(comparison_df)
        
        return {
            'comparison_table': comparison_data,
            'best_model_hr10': comparison_df.loc[comparison_df['HR@10'].idxmax(), 'Model'],
            'best_model_ndcg10': comparison_df.loc[comparison_df['NDCG@10'].idxmax(), 'Model'],
            'fastest_model': comparison_df.loc[comparison_df['Inference_Time_ms'].idxmin(), 'Model']
        }
    
    def generate_comparison_plots(self, comparison_df: pd.DataFrame):
        if not PLOTNINE_AVAILABLE:
            self.logger.warning("plotnine is unavailable; comparison plots were skipped")
            return
        
        metrics_plot = (ggplot(comparison_df.melt(id_vars=['Model'], 
                                                 value_vars=['HR@10', 'NDCG@10', 'MRR'],
                                                 var_name='Metric', value_name='Score'))
                       + aes(x='Model', y='Score', fill='Model')
                       + geom_col(position='dodge')
                       + facet_wrap('~Metric', scales='free_y')
                       + theme_minimal()
                       + theme(axis_text_x=element_text(rotation=45, hjust=1))
                       + labs(title='Model Performance Comparison',
                             x='Model', y='Score')
                       + scale_fill_brewer(type='qual', palette='Set2'))
        
        metrics_plot.save(os.path.join(self.output_dir, 'model_performance_comparison.png'), 
                         width=12, height=8, dpi=300)
        
        performance_plot = (ggplot(comparison_df)
                           + aes(x='Inference_Time_ms', y='HR@10', 
                                size='Memory_Usage_MB', color='Model')
                           + geom_point(alpha=0.7)
                           + theme_minimal()
                           + labs(title='Performance vs Accuracy Trade-off',
                                 x='Inference Time (ms)', y='HR@10',
                                 size='Memory Usage (MB)')
                           + scale_color_brewer(type='qual', palette='Set1'))
        
        performance_plot.save(os.path.join(self.output_dir, 'performance_accuracy_tradeoff.png'),
                             width=10, height=6, dpi=300)
    
    def ablation_study(self,
                      test_dataloader,
                      ablation_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        
        self.logger.info("Conducting ablation study...")
        
        ablation_results = {}
        
        for config in ablation_configs:
            config_name = config['name']
            model_modifications = config['modifications']
            
            self.logger.info(f"Testing ablation: {config_name}")
            
            modified_model = self._create_ablated_model(model_modifications)
            
            results = self.evaluate_model(modified_model, test_dataloader, config_name)
            ablation_results[config_name] = results
        
        self.generate_ablation_plots(ablation_results)
        
        return ablation_results
    
    def _create_ablated_model(self, modifications: Dict[str, Any]):
        raise NotImplementedError(
            "A valid MGSR ablation must change the architecture and retrain it. "
            "Freezing modules only at evaluation time would produce misleading results."
        )
    
    def generate_ablation_plots(self, ablation_results: Dict[str, Any]):
        if not PLOTNINE_AVAILABLE:
            self.logger.warning("plotnine is unavailable; ablation plots were skipped")
            return
        
        ablation_data = []
        
        for config_name, results in ablation_results.items():
            ranking_metrics = results['ranking_metrics']
            
            ablation_data.append({
                'Configuration': config_name,
                'HR@10': ranking_metrics.get('HR@10', 0.0),
                'NDCG@10': ranking_metrics.get('NDCG@10', 0.0),
                'MRR': ranking_metrics.get('MRR', 0.0)
            })
        
        ablation_df = pd.DataFrame(ablation_data)
        
        ablation_plot = (ggplot(ablation_df.melt(id_vars=['Configuration'],
                                               value_vars=['HR@10', 'NDCG@10', 'MRR'],
                                               var_name='Metric', value_name='Score'))
                        + aes(x='Configuration', y='Score', fill='Metric')
                        + geom_col(position='dodge')
                        + theme_minimal()
                        + theme(axis_text_x=element_text(rotation=45, hjust=1))
                        + labs(title='Ablation Study Results',
                              x='Configuration', y='Score')
                        + scale_fill_brewer(type='qual', palette='Set3'))
        
        ablation_plot.save(os.path.join(self.output_dir, 'ablation_study_results.png'),
                          width=12, height=8, dpi=300)
    
    def generate_evaluation_report(self):
        
        report_data = {
            'evaluation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'mgsr_results': self.evaluation_results.get('MGSR', {}),
            'baseline_results': self.baseline_results,
            'summary': self._generate_summary()
        }
        
        with open(os.path.join(self.output_dir, 'evaluation_report.json'), 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        self._generate_markdown_report(report_data)
    
    def _generate_summary(self) -> Dict[str, Any]:
        
        if 'MGSR' not in self.evaluation_results:
            return {}
        
        mgsr_metrics = self.evaluation_results['MGSR']['ranking_metrics']
        
        summary = {
            'key_metrics': {
                'HR@10': mgsr_metrics.get('HR@10', 0.0),
                'NDCG@10': mgsr_metrics.get('NDCG@10', 0.0),
                'MRR': mgsr_metrics.get('MRR', 0.0)
            }
        }
        
        return summary
    
    def _generate_markdown_report(self, report_data: Dict[str, Any]):
        
        markdown_content = f"""# MGSR Model Evaluation Report

Generated on: {report_data['evaluation_timestamp']}

## Executive Summary

MGSR (Multimodal Generative Sequential Recommendation with Adaptive Contrastive Learning) has been evaluated using the paper's ranking, diversity, novelty, and efficiency dimensions.

### Key Performance Metrics

| Metric | Value |
|--------|-------|
| HR@10 | {report_data['summary']['key_metrics']['HR@10']:.4f} |
| NDCG@10 | {report_data['summary']['key_metrics']['NDCG@10']:.4f} |
| MRR | {report_data['summary']['key_metrics']['MRR']:.4f} |

## Detailed Results

### Ranking Metrics
"""
        
        if report_data['mgsr_results']:
            ranking_metrics = report_data['mgsr_results']['ranking_metrics']
            for metric, value in ranking_metrics.items():
                markdown_content += f"- **{metric}**: {value:.4f}\n"
        
        markdown_content += """
### Performance Metrics
"""
        
        if report_data['mgsr_results']:
            perf_metrics = report_data['mgsr_results']['performance_metrics']
            for metric, value in perf_metrics.items():
                markdown_content += f"- **{metric}**: {value:.4f}\n"
        
        markdown_content += """
## Baseline Comparison

"""
        
        for model_name, results in report_data['baseline_results'].items():
            markdown_content += f"### {model_name}\n"
            ranking_metrics = results['ranking_metrics']
            markdown_content += f"- HR@10: {ranking_metrics.get('HR@10', 0.0):.4f}\n"
            markdown_content += f"- NDCG@10: {ranking_metrics.get('NDCG@10', 0.0):.4f}\n"
            markdown_content += f"- MRR: {ranking_metrics.get('MRR', 0.0):.4f}\n\n"
        
        markdown_content += """
## Visualizations

The following plots have been generated:
- Model Performance Comparison (`model_performance_comparison.png`)
- Performance vs Accuracy Trade-off (`performance_accuracy_tradeoff.png`)
- Ablation Study Results (`ablation_study_results.png`)

## Conclusions

The report records measured MGSR results without asserting dataset-independent target thresholds.
"""
        
        with open(os.path.join(self.output_dir, 'evaluation_report.md'), 'w') as f:
            f.write(markdown_content)
    
    def _compute_ndcg(self, predicted: List[int], relevant: List[int], k: int) -> float:
        relevant_set = set(relevant)
        
        dcg = 0.0
        for i, item in enumerate(predicted[:k]):
            if item in relevant_set:
                dcg += 1.0 / np.log2(i + 2)
        
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def _move_batch_to_device(self, batch_data: Dict[str, Any]) -> Dict[str, Any]:
        moved_batch = {}
        
        for key, value in batch_data.items():
            if isinstance(value, torch.Tensor):
                moved_batch[key] = value.to(self.device)
            elif isinstance(value, dict):
                moved_batch[key] = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                                   for k, v in value.items()}
            else:
                moved_batch[key] = value
        
        return moved_batch


MASREvaluator = MGSREvaluator
