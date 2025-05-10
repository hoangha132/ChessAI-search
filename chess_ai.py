import tkinter as tk
from tkinter import simpledialog, messagebox
from PIL import Image, ImageTk
import chess
import random
import time
import threading
import logging
from collections import defaultdict, deque

# Initialize Tkinter
root = tk.Tk()
root.title("Chess AI")

# Ask for difficulty levels
white_level = simpledialog.askinteger("White AI Level", "Select White AI (1-11):", minvalue=1, maxvalue=11, initialvalue=11)
black_level = simpledialog.askinteger("Black AI Level", "Select Black AI (1-11):", minvalue=1, maxvalue=11, initialvalue=7)

# Load piece images
PIECE_IMAGES = {}
PIECE_MAPPING = {
    'r': 'r.png', 'n': 'n.png', 'b': 'b.png', 'q': 'q.png', 'k': 'k.png', 'p': 'p.png',
    'R': 'RW.png', 'N': 'NW.png', 'B': 'BW.png', 'Q': 'QW.png', 'K': 'KW.png', 'P': 'PW.png'
}

for piece, filename in PIECE_MAPPING.items():
    try:
        img = Image.open(f"assets/{filename}").convert("RGBA").resize((60, 60))
        PIECE_IMAGES[piece] = ImageTk.PhotoImage(img)
    except (FileNotFoundError, AttributeError):
        PIECE_IMAGES[piece] = None

# Configure logging
logging.basicConfig(filename='chess_ai.log', level=logging.ERROR)

class ChessAI:
    def __init__(self):
        self.transposition_table = {}
        self.nodes_searched = 0
        self.max_time = 5.0
        self.start_time = 0
        self.killer_moves = defaultdict(set)
        self.history_moves = defaultdict(int)
        self.repetition_history = deque(maxlen=8)
        self.position_count = defaultdict(int)
        self.last_move_was_capture = False
        self.pv_table = {}
        self.level_adjustments = {
            1: {'depth': 1, 'time': 0.5, 'randomness': 0.9},
            2: {'depth': 1, 'time': 0.7, 'randomness': 0.7},
            3: {'depth': 2, 'time': 1.0, 'randomness': 0.5},
            4: {'depth': 2, 'time': 1.5, 'randomness': 0.3},
            5: {'depth': 3, 'time': 2.0, 'randomness': 0.2},
            6: {'depth': 3, 'time': 2.5, 'randomness': 0.1},
            7: {'depth': 4, 'time': 3.0, 'randomness': 0.05},
            8: {'depth': 5, 'time': 3.5, 'randomness': 0.02},
            9: {'depth': 6, 'time': 4.0, 'randomness': 0.01},
            10: {'depth': 7, 'time': 4.5, 'randomness': 0.005},
            11: {'depth': 8, 'time': 5.0, 'randomness': 0.001}
        }
        self.last_move_from = None
        self.last_move_to = None

    def get_move(self, board, level):
        """Get the best move for current position based on level"""
        try:
            self.nodes_searched = 0
            self.start_time = time.time()
            fen = board.board_fen()
            self.repetition_history.append(fen)
            self.position_count[fen] += 1
            
            # Check for threefold repetition
            if self.position_count[fen] >= 3:
                return None  # Draw by repetition
            
            settings = self.level_adjustments.get(level, {})
            self.max_time = settings.get('time', 2.0)
            
            if random.random() < settings.get('randomness', 0):
                move = self.random_move(board)
            elif level <= 3:
                move = self.random_move(board)
            elif level <= 6:
                move = self.quick_search(board, settings['depth'])
            else:
                move = self.iterative_deepening(board, settings['depth'])
            
            # Validate move is legal
            if move and move in board.legal_moves:
                self.last_move_was_capture = board.is_capture(move)
                self.last_move_from = move.from_square
                self.last_move_to = move.to_square
                return move
            else:
                # If invalid move, fall back to random legal move
                return random.choice(list(board.legal_moves))
        except Exception as e:
            logging.error(f"Error in get_move: {e}")
            return random.choice(list(board.legal_moves))

    def random_move(self, board):
        """Make random moves (for very low levels)"""
        return random.choice(list(board.legal_moves))

    def quick_search(self, board, depth):
        """Fast fixed-depth search with basic evaluation"""
        best_move = None
        best_score = -float('inf')
        
        for move in self.order_moves(board, list(board.legal_moves), 1):
            board.push(move)
            score = -self.evaluate(board)
            board.pop()
            
            if score > best_score:
                best_score = score
                best_move = move
                
            if time.time() - self.start_time > self.max_time/2:
                break
        
        return best_move or random.choice(list(board.legal_moves))

    def iterative_deepening(self, board, max_depth):
        """Time-limited iterative deepening search with PV tracking"""
        best_move = None
        best_score = -float('inf')
        self.pv_table = {}
        
        for depth in range(1, max_depth + 1):
            if time.time() - self.start_time > self.max_time:
                break
                
            # Extend search for critical positions
            effective_depth = depth
            if self.has_hanging_pieces(board):
                effective_depth += 1
                
            if self.game_phase(board) == 2:
                effective_depth += 1  # deepen in endgame

            move, score = self.alpha_beta(board, effective_depth)
            if score > best_score:
                best_score = score
                best_move = move
            
            if abs(score) > 9000:
                break
        
        return best_move or random.choice(list(board.legal_moves))

    def has_hanging_pieces(self, board):
        """Check if there are hanging pieces on the board"""
        for color in [chess.WHITE, chess.BLACK]:
            for piece_type in chess.PIECE_TYPES:
                for square in board.pieces(piece_type, color):
                    if board.is_attacked_by(not color, square) and not board.attackers(color, square):
                        return True
        return False

    def alpha_beta(self, board, depth, alpha=-float('inf'), beta=float('inf'), can_null=True, ply=0):
        """Enhanced alpha-beta with improved trade evaluation"""
        # Check for repetition
        fen = board.board_fen()
        if self.position_count.get(fen, 0) >= 2:
            return None, 0  # Prefer to avoid draws by repetition
        
        # Force capture of hanging pieces at root level
        if depth >= 1 and ply == 0:
            for move in board.legal_moves:
                if board.is_capture(move):
                    target = board.piece_at(move.to_square)
                    if target and not board.attackers(target.color, move.to_square):
                        return move, 9999 if board.turn == chess.WHITE else -9999
        
        tt_key = board._transposition_key()
        if tt_key in self.transposition_table:
            entry = self.transposition_table[tt_key]
            if entry['depth'] >= depth:
                if entry['flag'] == 'exact':
                    return entry['move'], entry['score']
                elif entry['flag'] == 'lower':
                    alpha = max(alpha, entry['score'])
                elif entry['flag'] == 'upper':
                    beta = min(beta, entry['score'])
                if alpha >= beta:
                    return entry['move'], entry['score']
        
        if depth == 0:
            return None, self.quiesce(board, alpha, beta)
            
        # Null move heuristic - only when not in danger
        if (depth >= 3 and can_null and not board.is_check() and 
            not self.has_hanging_pieces(board) and
            any(board.pieces(pt, board.turn) for pt in [chess.PAWN, chess.ROOK, chess.QUEEN])):
            board.push(chess.Move.null())
            _, null_score = self.alpha_beta(board, depth-3, -beta, -beta+1, False, ply+1)
            board.pop()
            if null_score >= beta:
                return None, beta
        
        best_move = None
        legal_moves = self.order_moves(board, list(board.legal_moves), depth)
        
        if not legal_moves:
            if board.is_check():
                return None, -99999 + ply if board.turn == chess.WHITE else 99999 - ply
            return None, 0
            
        pv_move = self.pv_table.get(board._transposition_key())
        if pv_move and pv_move in legal_moves:
            legal_moves.remove(pv_move)
            legal_moves.insert(0, pv_move)
            
        for i, move in enumerate(legal_moves):
            # Skip bad trades at higher depths
            if depth >= 3 and board.is_capture(move):
                target = board.piece_at(move.to_square)
                attacker = board.piece_at(move.from_square)
                if target and attacker and self.piece_value(target) < self.piece_value(attacker) * 0.85:
                    continue  # Skip bad trades
                    
            lmr_depth = depth - 1
            if depth >= 3 and i >= 4 and not board.is_capture(move) and not board.gives_check(move):
                lmr_depth = max(1, depth - 2)
                
            board.push(move)
            _, eval = self.alpha_beta(board, lmr_depth, -beta, -alpha, True, ply+1)
            board.pop()
            
            eval = -eval
            
            if eval >= beta:
                if not board.is_capture(move):
                    self.killer_moves[depth].add(move)
                    self.history_moves[move] += depth * depth
                
                self.transposition_table[tt_key] = {
                    'move': move,
                    'score': beta,
                    'depth': depth,
                    'flag': 'lower'
                }
                return move, beta
                
            if eval > alpha:
                alpha = eval
                best_move = move
                self.pv_table[board._transposition_key()] = move
                
            if time.time() - self.start_time > self.max_time:
                break
                
        flag = 'exact' if alpha == eval else ('upper' if best_move is None else 'lower')
        self.transposition_table[tt_key] = {
            'move': best_move,
            'score': alpha,
            'depth': depth,
            'flag': flag
        }
                
        return best_move, alpha

    def quiesce(self, board, alpha, beta):
        """Enhanced quiescence search with SEE pruning"""
        stand_pat = self.evaluate(board)
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat
        
        moves = [m for m in board.legal_moves if board.is_capture(m) or board.gives_check(m)]
        moves = sorted(moves, key=lambda m: self.see(board, m), reverse=True)
        
        for move in moves:
            # Skip bad trades in quiescence search
            if board.is_capture(move) and self.see(board, move) < -100:
                continue
                
            board.push(move)
            score = -self.quiesce(board, -beta, -alpha)
            board.pop()
            
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
                
        return alpha

    def see(self, board, move):
        """Static Exchange Evaluation with better trade evaluation"""
        if not board.is_capture(move):
            return 0
            
        gain = [self.piece_value(board.piece_at(move.to_square)) or 0]
        attacker = board.piece_at(move.from_square)
        gain[0] -= self.piece_value(attacker) if attacker else 0
        
        board.push(move)
        if board.is_check():
            board.pop()
            return 999
        board.pop()
        
        return gain[0]

    def order_moves(self, board, moves, depth):
        """Sophisticated move ordering with better trade evaluation"""
        def move_score(m):
            score = 0
            
            # 1. PV move from transposition table
            tt_move = self.transposition_table.get(board._transposition_key(), {}).get('move')
            if m == tt_move:
                return 9999
                
            # 2. Captures of hanging pieces (highest priority)
            if board.is_capture(m):
                target = board.piece_at(m.to_square)
                if target:
                    # Hanging piece (no defenders)
                    if not board.attackers(target.color, m.to_square):
                        score += 20000 + self.piece_value(target) * 10
                    # Normal capture - evaluate trade quality
                    else:
                        victim_val = self.piece_value(target)
                        attacker_val = self.piece_value(board.piece_at(m.from_square))
                        see_val = self.see(board, m)
                        
                        # Good trades
                        if see_val >= 0:
                            score += 10000 + victim_val * 10 - attacker_val
                        # Questionable trades
                        elif see_val > -100:
                            score += 5000 + see_val
                        # Bad trades (heavily penalized)
                        else:
                            score -= 3000
            
            # 3. Pawn pushes and promotions
            moving_piece = board.piece_at(m.from_square)
            if moving_piece and moving_piece.piece_type == chess.PAWN:
                # Pawn promotion
                if m.promotion:
                    score += 15000 + m.promotion * 10
                # Passed pawn push
                elif self.is_passed_pawn(board, m.from_square, board.turn):
                    rank = chess.square_rank(m.to_square)
                    score += rank * 20  # More advanced = better
            
            # 4. Defending threatened pieces
            if moving_piece:
                # Moving a threatened piece
                if board.is_attacked_by(not board.turn, m.from_square):
                    score += 4000
                # Moving to defend another piece
                for sq in board.attacks(m.to_square):
                    piece = board.piece_at(sq)
                    if piece and piece.color == board.turn and board.is_attacked_by(not board.turn, sq):
                        score += 3000
            
            # 5. Queen moves (prioritize active but safe queen play)
            if moving_piece and moving_piece.piece_type == chess.QUEEN:
                # Bonus for active queen
                score += 2000
                # Penalty if queen is moving into danger
                if board.is_attacked_by(not board.turn, m.to_square):
                    score -= 1000
            
            # 6. Killer moves
            for d in range(min(3, depth), 0, -1):
                if m in self.killer_moves[d]:
                    score += 8000 - (3 - d) * 1000
                    break
                    
            # 7. History heuristic
            score += self.history_moves.get(m, 0) // 10
            
            # 8. Castling
            if board.is_castling(m):
                score += 6000
                
            # 9. Checks
            if board.gives_check(m):
                # Check if the square we move to is under attack (unsafe)
                if board.is_attacked_by(not board.turn, m.to_square):
                    score -= 1500  # Penalize unsafe check
                else:
                    score += 2000  # Only reward safe checks
            
            # 10. Encourage defending hanging pieces
            if moving_piece and board.is_attacked_by(not board.turn, m.from_square):
                defenders = board.attackers(board.turn, m.from_square)
                if defenders:
                    score += 6000  
                
            return score
            
        return sorted(moves, key=move_score, reverse=True)

    def piece_value(self, piece):
        """Get the base value of a piece with phase consideration"""
        if not piece:
            return 0
            
        values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 0
        }
        return values.get(piece.piece_type, 0)

    def evaluate(self, board):
        """Comprehensive evaluation with improved trade and pawn evaluation"""
        if board.is_checkmate():
            return -99999 if board.turn == chess.WHITE else 99999
            
        if board.is_game_over():
            return 0
            
        score = 0
        phase = self.game_phase(board)
        
        # Material evaluation with hanging piece detection
        piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 0
        }
        
        # Evaluate hanging pieces and threats
        hanging_bonus = 0
        threat_score = 0
        queen_safety = 0
        pawn_structure = 0
        
        for color in [chess.WHITE, chess.BLACK]:
            sign = 1 if color == chess.WHITE else -1
            
            # Material count
            for piece_type, value in piece_values.items():
                white_count = len(board.pieces(piece_type, chess.WHITE))
                black_count = len(board.pieces(piece_type, chess.BLACK))
                score += (white_count - black_count) * value
            
            # Check all pieces for hanging status
            for piece_type in chess.PIECE_TYPES:
                for square in board.pieces(piece_type, color):
                    attackers = board.attackers(not color, square)
                    defenders = board.attackers(color, square)
                    
                    # If piece is under attack and undefended
                    if attackers and not defenders:
                        hanging_value = piece_values[piece_type] * 0.5  # 50% of piece value
                        hanging_bonus -= hanging_value * sign
                        
                    # If piece is attacking more valuable enemy piece
                    for target_sq in board.attacks(square):
                        target_piece = board.piece_at(target_sq)
                        if target_piece and target_piece.color != color:
                            if piece_values[piece_type] < piece_values[target_piece.piece_type]:
                                threat_score += (piece_values[target_piece.piece_type] - piece_values[piece_type]) * 0.3 * sign

            # Queen safety evaluation
            queens = list(board.pieces(chess.QUEEN, color))
            if queens:
                queen_sq = queens[0]
                # Penalize queen being attacked
                if board.is_attacked_by(not color, queen_sq):
                    queen_safety -= 250 * sign  # Increased penalty
                # Bonus for active queen
                queen_activity = len(board.attacks(queen_sq))
                queen_safety += queen_activity * 10 * sign  # Reduced bonus
                # Bonus for central control
                if self.square_center_proximity(queen_sq) > 3:
                    queen_safety += 50 * sign
                # Penalty for undeveloped queen in opening
                if phase == 0 and chess.square_rank(queen_sq) == (0 if color == chess.WHITE else 7):
                    queen_safety -= 50 * sign

            # Piece coordination evaluation
            knights = list(board.pieces(chess.KNIGHT, color))
            bishops = list(board.pieces(chess.BISHOP, color))
            
            # Bonus for bishop pair
            if len(bishops) >= 2:
                score += 30 * sign
                
            # Penalty for blocking own pieces
            for knight_sq in knights:
                if board.piece_at(knight_sq) and board.piece_at(knight_sq).piece_type == chess.PAWN:
                    score -= 10 * sign
            
            if self.position_count[board.board_fen()] >= 2:
                score -= 100 * (1 if board.turn == chess.WHITE else -1)

            # Development score
            if phase == 0:  # Opening phase
                # Bonus for castling
                if board.has_castling_rights(color):
                    score += 20 * sign
                elif self.has_castled(board, color):
                    score += 40 * sign
                
                # Penalty for undeveloped pieces
                back_rank = chess.BB_RANK_1 if color == chess.WHITE else chess.BB_RANK_8
                undeveloped = board.occupied_co[color] & back_rank & ~board.kings & ~board.queens
                score -= chess.popcount(undeveloped) * 15 * sign

            # Pawn structure evaluation
            pawns = board.pieces(chess.PAWN, color)
            files = [chess.square_file(p) for p in pawns]
            
            # Doubled pawns
            doubled = sum(files.count(f) > 1 for f in set(files))
            pawn_structure -= 15 * doubled * sign
            
            # Isolated pawns
            isolated = 0
            for f in set(files):
                if not any(abs(f - f2) == 1 for f2 in set(files) if f2 != f):
                    isolated += 1
            pawn_structure -= 10 * isolated * sign
            
            # Passed pawns
            for pawn_sq in pawns:
                if self.is_passed_pawn(board, pawn_sq, color):
                    rank = chess.square_rank(pawn_sq)
                    advance_bonus = (rank if color == chess.WHITE else 7-rank) * 30
                    pawn_structure += advance_bonus * sign

        score += hanging_bonus + threat_score + queen_safety + pawn_structure

        # King safety
        for color in [chess.WHITE, chess.BLACK]:
            sign = 1 if color == chess.WHITE else -1
            king_sq = board.king(color)
            if king_sq:
                # Penalize unsafe kings
                attackers = board.attackers(not color, king_sq)
                score += (-20 if color == chess.WHITE else 20) * len(attackers)
            if phase == 2:
                dist_to_center = 3.5 - self.square_center_proximity(king_sq)
                score += 40 * (3.5 - dist_to_center) * sign  # closer to center is better
        
        # Mobility
        mobility = len(list(board.legal_moves))
        score += mobility * (2 if phase == 2 else 1) * (1 if board.turn == chess.WHITE else -1)
        
        return score if board.turn == chess.WHITE else -score

    def has_castled(self, board, color):
        """Check if the king has castled"""
        king_sq = board.king(color)
        if color == chess.WHITE:
            return king_sq in [chess.G1, chess.C1]
        else:
            return king_sq in [chess.G8, chess.C8]

    def is_passed_pawn(self, board, pawn_sq, color):
        """Check if a pawn is passed"""
        file = chess.square_file(pawn_sq)
        rank = chess.square_rank(pawn_sq)
        
        if color == chess.WHITE:
            enemy_pawns = board.pieces(chess.PAWN, chess.BLACK)
            for f in [file-1, file, file+1]:
                if f < 0 or f > 7:
                    continue
                for r in range(rank+1, 8):
                    if chess.square(f, r) in enemy_pawns:
                        return False
        else:
            enemy_pawns = board.pieces(chess.PAWN, chess.WHITE)
            for f in [file-1, file, file+1]:
                if f < 0 or f > 7:
                    continue
                for r in range(rank-1, -1, -1):
                    if chess.square(f, r) in enemy_pawns:
                        return False
        return True

    def square_center_proximity(self, square):
        """Calculate how close a square is to the center (d4,d5,e4,e5)"""
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        center_files = [3, 4]  # d and e files
        center_ranks = [3, 4]  # 4th and 5th ranks
        
        file_dist = min(abs(file - c) for c in center_files)
        rank_dist = min(abs(rank - c) for c in center_ranks)
        
        return 8 - (file_dist + rank_dist)  # Higher is better

    def game_phase(self, board):
        """Determine game phase (0=opening, 1=midgame, 2=endgame)"""
        total_material = sum(1 for _ in board.piece_map().values())
        if total_material > 30: return 0
        if total_material > 16: return 1
        return 2

class ChessGUI:
    def __init__(self, root, white_level, black_level):
        self.root = root
        self.board = chess.Board()
        self.ai = ChessAI()
        self.white_level = white_level
        self.black_level = black_level
        self.move_count = 0
        self.last_move_from = None
        self.last_move_to = None
        self.move_history = []
        
        # Setup GUI
        self.canvas = tk.Canvas(root, width=600, height=650, bg="white")
        self.canvas.pack()
        
        # Add level display
        self.level_display = tk.Label(root, 
            text=f"White (Level {white_level}) vs Black (Level {black_level})",
            font=("Arial", 14))
        self.level_display.pack()
        
        # Add move delay slider
        self.delay_var = tk.DoubleVar(value=1.0)
        tk.Scale(root, from_=0, to=3, resolution=0.1, orient=tk.HORIZONTAL,
                label="AI Move Delay (seconds)", variable=self.delay_var).pack()
        
        # Add move history display
        self.move_history_label = tk.Label(root, text="Move History:", font=("Arial", 12))
        self.move_history_label.pack()
        self.move_history_text = tk.Text(root, height=6, width=40, font=("Arial", 10))
        self.move_history_text.pack()
        
        # Start game
        self.update_board()
        self.ai_move("white")

    def update_board(self):
        """Draw the current board state with move indicators"""
        self.canvas.delete("all")
        colors = ["#F0D9B5", "#B58863"]
        square_size = 75
        
        # Draw squares
        for row in range(8):
            for col in range(8):
                color = colors[(row + col) % 2]
                x1, y1 = col * square_size, row * square_size
                x2, y2 = x1 + square_size, y1 + square_size
                
                # Highlight last move
                current_square = chess.square(col, 7 - row)
                if current_square == self.ai.last_move_from or current_square == self.ai.last_move_to:
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill="#FFFF00", outline="#FFFF00")
                else:
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline=color)
                
                # Draw pieces
                piece = self.board.piece_at(chess.square(col, 7 - row))
                if piece:
                    if PIECE_IMAGES[piece.symbol()]:
                        self.canvas.create_image(x1 + square_size//2, y1 + square_size//2, 
                                               image=PIECE_IMAGES[piece.symbol()], 
                                               anchor=tk.CENTER)
                    else:
                        self.canvas.create_text(x1 + square_size//2, y1 + square_size//2, 
                                              text=piece.symbol(), 
                                              font=("Arial", 32))
        
        # Check game over
        if self.board.is_game_over():
            result = self.board.result()
            if result == "1-0":
                messagebox.showinfo("Game Over", f"White (Level {self.white_level}) wins!")
            elif result == "0-1":
                messagebox.showinfo("Game Over", f"Black (Level {self.black_level}) wins!")
            else:
                messagebox.showinfo("Game Over", "Draw!")
            return False
        return True

    def ai_move(self, color):
        """Make AI move for specified color with delay"""
        if not self.update_board():
            return
            
        level = self.white_level if color == "white" else self.black_level
        
        def get_move():
            try:
                start_time = time.time()
                move = self.ai.get_move(self.board, level)
                
                # Check for draw by repetition
                if move is None:
                    messagebox.showinfo("Game Over", "Draw by repetition!")
                    return
                
                # Validate move before applying
                if move not in self.board.legal_moves:
                    logging.error(f"Illegal move attempted: {move} in {self.board.fen()}")
                    move = random.choice(list(self.board.legal_moves))
                
                elapsed = time.time() - start_time
                
                # Ensure minimum delay
                min_delay = max(0, self.delay_var.get() - elapsed)
                if min_delay > 0:
                    time.sleep(min_delay)
                    
                self.board.push(move)
                self.move_count += 1
                self.last_move_from = move.from_square
                self.last_move_to = move.to_square
                
                # Update move history
                try:
                    move_san = self.board.san(move)
                    turn = "White" if color == "white" else "Black"
                    move_text = f"{turn}: {move_san}"
                    self.move_history.append(move_text)
                    self.update_move_history()
                except Exception as e:
                    logging.error(f"Error recording move: {e}")
                    self.move_history.append(f"{color} made a move (error in notation)")
                    self.update_move_history()
                    
                self.root.after(10, lambda: self.ai_move("black" if color == "white" else "white"))
            except Exception as e:
                logging.error(f"Error in AI move: {e}")
                # Reset board if in illegal state
                self.board = chess.Board()
                self.ai_move("white")
        
        threading.Thread(target=get_move, daemon=True).start()

    def update_move_history(self):
        """Update the move history display"""
        self.move_history_text.delete(1.0, tk.END)
        for i, move in enumerate(self.move_history):
            self.move_history_text.insert(tk.END, f"{i+1}. {move}\n")

if __name__ == "__main__":
    ChessGUI(root, white_level, black_level)
    root.mainloop() 