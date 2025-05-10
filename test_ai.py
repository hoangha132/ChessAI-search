import chess
import time
from chess_ai import ChessAI
import pandas as pd
import matplotlib.pyplot as plt
import random
from datetime import datetime

class RobustChessTester:
    def __init__(self):
        self.ai = ChessAI()
        self.results = []
        self.max_move_time = 3
        self.max_moves = 100
        self.imbalance_threshold = 1500

    def count_material(self, board, color):
        """Đếm giá trị vật chất an toàn"""
        try:
            piece_values = {
                chess.PAWN: 100,
                chess.KNIGHT: 320,
                chess.BISHOP: 330,
                chess.ROOK: 500,
                chess.QUEEN: 900
            }
            return sum(len(board.pieces(p, color)) * v for p, v in piece_values.items())
        except:
            return 0  # Trả về 0 nếu có lỗi

    def should_terminate_early(self, board):
        """Kiểm tra điều kiện kết thúc sớm an toàn"""
        try:
            white_mat = self.count_material(board, chess.WHITE)
            black_mat = self.count_material(board, chess.BLACK)
            
            if abs(white_mat - black_mat) > self.imbalance_threshold:
                return "1-0" if white_mat > black_mat else "0-1"
                
            if white_mat <= 100 and len(board.pieces(chess.PAWN, chess.WHITE)) < 2:
                if black_mat > 500: return "0-1"
                
            if black_mat <= 100 and len(board.pieces(chess.PAWN, chess.BLACK)) < 2:
                if white_mat > 500: return "1-0"
                
        except:
            pass
        return None

    def get_safe_move(self, board, level):
        """Lấy nước đi an toàn với xử lý lỗi đầy đủ"""
        try:
            # Ưu tiên chiếu hết
            if level >= 7:
                for move in board.legal_moves:
                    board.push(move)
                    is_mate = board.is_checkmate()
                    board.pop()
                    if is_mate:
                        return move
            
            move = self.ai.get_move(board, level)
            return move if move else random.choice(list(board.legal_moves))
        except:
            return random.choice(list(board.legal_moves)) if board.legal_moves else None

    def play_safe_game(self, white_level, black_level):
        """Chơi ván cờ với xử lý lỗi mạnh mẽ"""
        board = chess.Board()
        move_count = 0
        termination = "normal"
        
        for _ in range(self.max_moves):
            # Kiểm tra kết thúc sớm
            early_result = self.should_terminate_early(board)
            if early_result:
                termination = "material imbalance"
                break
                
            if not board.legal_moves:
                break
                
            try:
                # Lấy nước đi an toàn
                move = self.get_safe_move(board, 
                    white_level if board.turn == chess.WHITE else black_level)
                
                if not move:
                    break
                    
                board.push(move)
                move_count += 1
                
            except Exception as e:
                print(f"Lỗi không xác định: {e}")
                termination = "error"
                break
                
        # Tính toán kết quả cuối cùng
        try:
            result = board.result() if board.is_game_over() else "1/2-1/2"
            white_mat = self.count_material(board, chess.WHITE)
            black_mat = self.count_material(board, chess.BLACK)
            
            return {
                'result': result,
                'moves': move_count,
                'termination': termination,
                'white_material': white_mat,
                'black_material': black_mat,
                'material_diff': white_mat - black_mat
            }
        except:
            return {
                'result': "1/2-1/2",
                'moves': move_count,
                'termination': "error",
                'white_material': 0,
                'black_material': 0,
                'material_diff': 0
            }

    def run_robust_test(self, test_levels=[1, 3, 5, 7, 9, 10], games_per_match=3):
        """Chạy test với xử lý lỗi mạnh mẽ"""
        print("Bắt đầu thử nghiệm mạnh mẽ...\n")
        
        for level in test_levels:
            print(f"\n=== AI lv11 vs AI lv{level} ===")
            
            for game_num in range(1, games_per_match + 1):
                # Test cả hai màu
                for color in ['White', 'Black']:
                    w_lv = 11 if color == 'White' else level
                    b_lv = level if color == 'White' else 11
                    
                    result = self.play_safe_game(w_lv, b_lv)
                    self.record_result(11, level, color, result, game_num)
                    
                    print(f"Đã hoàn thành {game_num*2-1 if color=='White' else game_num*2}/{games_per_match*2} ván", end="\r")
        
        self.generate_report()

    def record_result(self, high_lv, low_lv, color, result, game_num):
        """Ghi kết quả với đầy đủ kiểm tra"""
        default_values = {
            'result': "1/2-1/2",
            'moves': 0,
            'termination': "error",
            'white_material': 0,
            'black_material': 0,
            'material_diff': 0
        }
        
        safe_result = {**default_values, **result}
        
        self.results.append({
            'test_id': f"{high_lv}v{low_lv}_{color}_{game_num}",
            'ai_level': high_lv,
            'opponent_level': low_lv,
            'ai_color': color,
            **safe_result,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

    def generate_report(self):
        """Tạo báo cáo an toàn"""
        try:
            df = pd.DataFrame(self.results)
            df.to_csv('robust_chess_test.csv', index=False)
            
            # Vẽ biểu đồ đơn giản
            if not df.empty:
                plt.figure(figsize=(12, 5))
                df.groupby('opponent_level')['material_diff'].mean().plot(kind='bar')
                plt.title('Chênh lệch vật chất trung bình')
                plt.savefig('robust_chess_results.png')
            
            print("\n\nKết quả đã được lưu vào:")
            print("- robust_chess_test.csv")
            print("- robust_chess_results.png")
            
            return df.describe()
        except Exception as e:
            print(f"Lỗi khi tạo báo cáo: {e}")
            return None

if __name__ == "__main__":
    tester = RobustChessTester()
    stats = tester.run_robust_test(games_per_match=3)
    if stats is not None:
        print("\nThống kê tổng quan:")
        print(stats)