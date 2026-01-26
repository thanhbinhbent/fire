from typing import Dict, Optional, List
import concurrent.futures
from common import modeling
from eval.fire import query_serper


class FastVerifier:
    def __init__(self, model: modeling.Model, serper_api_key: str):
        self.model = model
        self.serper = query_serper.VietnameseSerperAPI(serper_api_key)
        self._cache = {}
        
    def verify_fast(self, claim: str) -> Dict:
        import time
        start = time.time()
        
        if claim in self._cache:
            result = self._cache[claim].copy()
            result['latency'] = time.time() - start
            result['from_cache'] = True
            return result
        
        analysis = self._analyze_and_decide(claim)
        
        queries = analysis.get('suggested_queries', [claim[:50]])
        queries = queries[:1] if analysis.get('is_basic_fact') else queries[:2]
        
        search_results = self._parallel_search(queries)
        final_result = self._quick_final_answer(claim, search_results, analysis)
        
        final_result['latency'] = time.time() - start
        final_result['fast_path'] = analysis.get('is_basic_fact', False)
        final_result['searches'] = len(search_results)
        
        self._cache[claim] = final_result
        return final_result
    
    def _analyze_and_decide(self, claim: str) -> Dict:
        prompt = f"""Phân tích nhận định này và đề xuất các truy vấn tìm kiếm.

Nhận định: "{claim}"

Trả về JSON:
{{
    "is_basic_fact": true/false,  // Sự thật phổ biến (địa lý, toán học, lịch sử)
    "verdict": "True"/"False"/"Unknown",  // Đánh giá ban đầu (có thể điều chỉnh sau khi có bằng chứng)
    "confidence": 0.0-1.0,  // Độ tin cậy ban đầu
    "reasoning": "giải thích ngắn gọn bằng tiếng Việt",
    "suggested_queries": ["truy vấn 1", "truy vấn 2"]  // LUÔN cung cấp 1-2 truy vấn tìm kiếm
}}

Quy tắc:
- is_basic_fact: TRUE cho địa lý, toán học, lịch sử nổi tiếng, định nghĩa
- LUÔN cung cấp suggested_queries (ngay cả với sự thật phổ biến - để có bằng chứng trích dẫn)
- Sự thật phổ biến: 1 truy vấn đơn giản
- Nhận định phức tạp: 2 truy vấn tập trung (3-5 từ mỗi truy vấn)
- Chỉ trích xuất các từ khóa chính, tránh câu đầy đủ

Chỉ trả về JSON hợp lệ."""

        try:
            response, _ = self.model.generate(prompt)
            from common.utils import extract_json_from_output
            analysis = extract_json_from_output(response)
            
            if not analysis:
                return {
                    'is_basic_fact': False,
                    'needs_search': True,
                    'suggested_queries': [claim[:50]]
                }
            return analysis
        except Exception as e:
            print(f"Analysis error: {e}")
            return {
                'is_basic_fact': False,
                'needs_search': True,
                'suggested_queries': [claim[:50]]
            }
    
    def _parallel_search(self, queries: List[str]) -> List[Dict]:
        if not queries:
            return []
        
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(queries)) as executor:
            future_to_query = {executor.submit(self._single_search, query): query for query in queries}
            for future in concurrent.futures.as_completed(future_to_query):
                try:
                    result = future.result(timeout=5.0)
                    if result:
                        results.extend(result)
                except Exception as e:
                    print(f"Search failed: {e}")
        return results
    
    def _single_search(self, query: str) -> List[Dict]:
        if not query or not query.strip():
            return []
            
        try:
            base_api = self.serper.base_api
            raw_results = base_api._google_serper_api_results(query, search_type='search', num=3)
            result_key = base_api.result_key_for_type['search']
            
            if result_key not in raw_results:
                return []
            
            return [
                {
                    'title': item.get('title', 'Kết quả tìm kiếm'),
                    'snippet': item.get('snippet', ''),
                    'link': item.get('link', '')
                }
                for item in raw_results[result_key][:3]
            ]
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    def _quick_final_answer(self, claim: str, search_results: List[Dict], analysis: Dict) -> Dict:
        if search_results:
            evidence_text = "\n\n".join([
                f"Nguồn {i+1}: {r.get('title', 'Kết quả tìm kiếm')}\nURL: {r.get('link', '')}\nNội dung: {r.get('snippet', '')[:300]}"
                for i, r in enumerate(search_results[:3])
            ])
        else:
            evidence_text = "Không có kết quả tìm kiếm. Sử dụng kiến thức nội bộ một cách cẩn thận."
        
        prompt = f"""Kiểm tra tính chính xác của nhận định này dựa trên bằng chứng được cung cấp.

Nhận định: "{claim}"

Bằng chứng tìm kiếm:
{evidence_text}

Phân tích bằng chứng và cung cấp:
1. Trích dẫn các sự thật cụ thể từ bằng chứng hỗ trợ hoặc bác bỏ nhận định
2. Giải thích lý luận của bạn từng bước bằng tiếng Việt
3. Đưa ra phán quyết cuối cùng với độ tin cậy

Trả về JSON:
{{
    "verdict": "True"/"False"/"Not Enough Info",
    "confidence": 0.0-1.0,
    "reasoning": "Giải thích chi tiết bằng tiếng Việt, trích dẫn bằng chứng. Ví dụ: 'Theo Nguồn 1, [trích dẫn]. Điều này [hỗ trợ/mâu thuẫn] với nhận định vì...'"
}}

Quy tắc:
- True: Bằng chứng rõ ràng hỗ trợ nhận định
- False: Bằng chứng rõ ràng mâu thuẫn với nhận định
- Not Enough Info: Bằng chứng không đủ hoặc mâu thuẫn
- LUÔN trích dẫn nguồn nào bạn đã sử dụng trong lý luận
- Trích dẫn các sự thật cụ thể, không chỉ tóm tắt
- Giải thích bằng tiếng Việt

Chỉ trả về JSON hợp lệ."""

        try:
            response, _ = self.model.generate(prompt)
            from common.utils import extract_json_from_output
            result = extract_json_from_output(response)
            
            if not result:
                return {
                    'verdict': 'Not Enough Info',
                    'confidence': 0.3,
                    'reasoning': 'Không thể phân tích phản hồi từ LLM',
                    'sources': []
                }
            
            sources = [
                {
                    'url': r.get('link', ''),
                    'title': r.get('title', 'Kết quả tìm kiếm'),
                    'snippet': r.get('snippet', ''),
                    'credibility': 0.7,
                    'relevance': 0.8
                }
                for r in search_results[:3] if r.get('link')
            ]
            
            result['sources'] = sources
            return result
        except Exception as e:
            print(f"Final answer error: {e}")
            return {
                'verdict': 'Not Enough Info',
                'confidence': 0.3,
                'reasoning': f'Lỗi: {e}',
                'sources': []
            }
    
    def clear_cache(self):
        self._cache.clear()
    
    def get_cache_size(self) -> int:
        return len(self._cache)


def verify_claim_fast(
    claim: str,
    model: modeling.Model,
    serper_api_key: str,
    use_cache: bool = True
) -> Dict:
    verifier = FastVerifier(model, serper_api_key)
    if not use_cache:
        verifier.clear_cache()
    return verifier.verify_fast(claim)
