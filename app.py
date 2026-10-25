"""AirSketch AI - Pygame front end.  Run:  python app.py"""
from __future__ import annotations

import math
import os
import sys
import threading
import time
from typing import Callable, List, Optional, Tuple

import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import pygame

import challenges as ch
from camera import CameraCapture
from drawing_canvas import BRUSH_SIZES, CANVAS_H, CANVAS_W, COLORS, DrawingCanvas, normalized_to_canvas
from game_logic import Game, GameState, grade_for_score
from hand_tracking import FIST, OPEN, PEACE, POINT, HandState, HandTrackingWorker, HoldDetector
from image_evaluator import EvaluationResult, ImageEvaluator

W, H = 1360, 780
BG, PANEL, INK = (24, 27, 46), (36, 40, 66), (240, 242, 255)
MUTED, ACCENT, ACCENT2 = (150, 156, 190), (255, 196, 61), (94, 214, 190)
GOOD, BAD = (110, 220, 130), (255, 107, 107)
CANVAS_POS = (360, 110)
PREVIEW = (320, 240)


# ------------------------------------------------------------------ small UI helpers
class Button:
    def __init__(self, rect, label: str, action: Callable[[], None], color=ACCENT, text_color=(30, 30, 50)):
        self.rect, self.label, self.action = pygame.Rect(rect), label, action
        self.color, self.text_color, self.selected, self.enabled = color, text_color, False, True

    def draw(self, surf, font, mouse):
        hover = self.enabled and self.rect.collidepoint(mouse)
        col = self.color if self.enabled else (70, 74, 100)
        if hover:
            col = tuple(min(255, c + 25) for c in col)
        pygame.draw.rect(surf, col, self.rect, border_radius=12)
        if self.selected:
            pygame.draw.rect(surf, INK, self.rect, 3, border_radius=12)
        txt = font.render(self.label, True, self.text_color if self.enabled else MUTED)
        surf.blit(txt, txt.get_rect(center=self.rect.center))

    def click(self, pos) -> bool:
        if self.enabled and self.rect.collidepoint(pos):
            self.action()
            return True
        return False


def pil_to_surface(img, size: Optional[Tuple[int, int]] = None) -> pygame.Surface:
    img = img.convert("RGB")
    if size:
        img = img.resize(size)
    return pygame.image.frombytes(img.tobytes(), img.size, "RGB")


def wrap(text: str, font, width: int) -> List[str]:
    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if font.size(trial)[0] <= width:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    return lines + ([cur] if cur else [])


# ------------------------------------------------------------------ application
class App:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("AirSketch AI")
        self.screen = pygame.display.set_mode((W, H))
        self.clock = pygame.time.Clock()
        names = "segoeui,helveticaneue,arial"
        self.f_title = pygame.font.SysFont(names, 54, bold=True)
        self.f_h = pygame.font.SysFont(names, 30, bold=True)
        self.f_b = pygame.font.SysFont(names, 21)
        self.f_s = pygame.font.SysFont(names, 17)
        self.f_big = pygame.font.SysFont(names, 96, bold=True)
        self.f_timer = pygame.font.SysFont(names, 72, bold=True)

        self.game = Game()
        self.canvas = DrawingCanvas()
        self.evaluator = ImageEvaluator()
        self.reference_img = None
        self.reference_surf: Optional[pygame.Surface] = None
        self.result: Optional[EvaluationResult] = None
        self.final_drawing = None
        self.final_surf: Optional[pygame.Surface] = None
        self._eval_lock = threading.Lock()
        self._pending: Optional[EvaluationResult] = None
        self._candidates = None

        self.input_mode = "webcam"
        self.camera: Optional[CameraCapture] = None
        self.worker: Optional[HandTrackingWorker] = None
        self.camera_msg = ""
        self.pen_enabled = True
        self.hand = HandState()
        self.clear_hold = HoldDetector(OPEN, 1.0)
        self.toast, self.toast_until = "", 0.0
        self.menu_buttons: List[Button] = []
        self.play_buttons: List[Button] = []
        self.result_buttons: List[Button] = []
        self.swatches: List[Tuple[pygame.Rect, tuple]] = []
        self.brush_rects: List[Tuple[pygame.Rect, int]] = []
        self._build_ui()
        ch.ensure_assets()
        self.start_camera()

    # ------------------------------------------------------------ UI construction
    def _build_ui(self):
        self.menu_buttons = []
        for i, d in enumerate(ch.DIFFICULTIES):
            b = Button((W // 2 - 330 + i * 230, 470, 200, 56), d.capitalize(), lambda d=d: self.game.set_difficulty(d))
            self.menu_buttons.append(b)
        self.btn_start = Button((W // 2 - 160, 560, 320, 70), "Start Game", self.start_game, ACCENT2)
        self.btn_input = Button((W // 2 - 160, 650, 320, 46), "", self.toggle_input, PANEL, INK)
        self.btn_submit = Button((20, 545, 320, 52), "Submit drawing  (Enter)", self.submit, GOOD)
        self.btn_clear = Button((20, 607, 154, 46), "Clear (C)", self.clear_canvas, ACCENT)
        self.btn_skip = Button((186, 607, 154, 46), "Skip (S)", self.skip, BAD)
        self.btn_pen = Button((20, 665, 320, 46), "", self.toggle_pen, PANEL, INK)
        self.play_buttons = [self.btn_submit, self.btn_clear, self.btn_skip, self.btn_pen]
        self.result_buttons = [
            Button((W // 2 - 480, 690, 300, 56), "Play Again (same)", self.play_again, ACCENT2),
            Button((W // 2 - 150, 690, 300, 56), "New Challenge", self.new_challenge, ACCENT),
            Button((W // 2 + 180, 690, 300, 56), "Menu", self.game.to_menu, PANEL, INK),
        ]
        x0 = 1020
        for i, (_, rgb) in enumerate(COLORS.items()):
            self.swatches.append((pygame.Rect(x0 + i * 52, 400, 42, 42), rgb))
        for i, s in enumerate(BRUSH_SIZES):
            self.brush_rects.append((pygame.Rect(x0 + i * 70, 480, 60, 50), s))

    # ------------------------------------------------------------ camera
    def start_camera(self):
        self.stop_camera()
        self.camera_msg = ""
        if self.input_mode != "webcam":
            return
        self.camera = CameraCapture()
        self.camera.start()
        self.worker = HandTrackingWorker(self.camera)
        self.worker.start()

    def stop_camera(self):
        for t in (self.worker, self.camera):
            if t:
                t.stop()
        self.camera = self.worker = None

    def check_camera_health(self):
        """Fall back to the mouse if the webcam or MediaPipe fails; never crash."""
        if self.input_mode != "webcam":
            return
        err = (self.camera and self.camera.error) or (self.worker and self.worker.error)
        if err:
            self.camera_msg = str(err)
            self.input_mode = "mouse"
            self.stop_camera()
            self.show_toast("Webcam unavailable - switched to mouse drawing.", 5)

    def toggle_input(self):
        self.input_mode = "mouse" if self.input_mode == "webcam" else "webcam"
        self.start_camera()

    # ------------------------------------------------------------ actions
    def show_toast(self, msg: str, secs: float = 2.5):
        self.toast, self.toast_until = msg, time.monotonic() + secs

    def _load_round(self):
        c = self.game.challenge
        self.reference_img = ch.load_reference(c)
        self.reference_surf = pil_to_surface(self.reference_img, (300, 300))
        self.canvas = DrawingCanvas()
        self.result = self.final_surf = self.final_drawing = None
        self.pen_enabled = True

    def start_game(self):
        self.game.start()
        self._load_round()

    def play_again(self):
        if self.game.play_again():
            self._load_round()

    def new_challenge(self):
        if self.game.new_challenge():
            self._load_round()

    def skip(self):
        if self.game.skip():
            self._load_round()
            self.show_toast("Challenge skipped")

    def clear_canvas(self):
        if self.game.can_draw:
            self.canvas.clear()

    def toggle_pen(self):
        self.pen_enabled = not self.pen_enabled
        self.canvas.lift()

    def submit(self):
        if self.game.state is not GameState.PLAYING:
            return
        if self.canvas.is_empty():
            self.show_toast("Draw something first!")
            return
        if self.game.submit():
            self.begin_evaluation()

    def begin_evaluation(self):
        """Freeze the canvas and evaluate on a worker thread so the UI stays responsive."""
        self.canvas.frozen = True
        self.canvas.lift()
        self.final_drawing = self.canvas.to_pil()
        self.final_surf = pil_to_surface(self.final_drawing, (360, 270))
        reference, name = self.reference_img, self.game.challenge.name
        if self._candidates is None:
            self._candidates = ch.all_references()

        def work():
            try:
                res = self.evaluator.evaluate(reference, self.final_drawing, name, self._candidates)
            except Exception as exc:  # last-resort guard: the game must never crash here
                res = EvaluationResult(0.0, error=f"Evaluation failed unexpectedly: {type(exc).__name__}")
            with self._eval_lock:
                self._pending = res

        threading.Thread(target=work, daemon=True).start()

    def poll_evaluation(self):
        with self._eval_lock:
            res, self._pending = self._pending, None
        if res is None or self.game.state is not GameState.EVALUATING:
            return
        self.result = res
        if res.ok:
            self.game.record_result(res.score, res.method)
        else:
            self.game.abort_evaluation()

    # ------------------------------------------------------------ per-frame input
    def update_drawing(self):
        g = self.game
        if g.tick():
            self.begin_evaluation()
        if not g.can_draw:
            self.canvas.lift()
            return
        self.canvas.frozen = False
        if self.input_mode == "webcam" and self.worker:
            self.hand = self.worker.state()
            h = self.hand
            if h.tip is None:
                self.canvas.lift()                  # hand left the frame -> end stroke
                self.clear_hold.update("none", time.monotonic())
                return
            if self.clear_hold.update(h.gesture, time.monotonic()):
                self.canvas.clear()
                self.show_toast("Canvas cleared")
            if h.gesture == POINT and self.pen_enabled:
                self.canvas.add_point(*normalized_to_canvas(*h.tip))
            else:
                self.canvas.lift()
        else:
            mx, my = pygame.mouse.get_pos()
            rx, ry = mx - CANVAS_POS[0], my - CANVAS_POS[1]
            inside = 0 <= rx < CANVAS_W and 0 <= ry < CANVAS_H
            if pygame.mouse.get_pressed()[0] and inside and self.pen_enabled:
                self.canvas.add_point(rx, ry)
            else:
                self.canvas.lift()

    def handle_event(self, e):
        if e.type == pygame.QUIT:
            return False
        if e.type == pygame.KEYDOWN:
            if e.key == pygame.K_ESCAPE:
                return False
            if self.game.state is GameState.PLAYING:
                if e.key == pygame.K_SPACE:
                    self.toggle_pen()
                elif e.key == pygame.K_c:
                    self.clear_canvas()
                elif e.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    self.submit()
                elif e.key == pygame.K_s:
                    self.skip()
            elif self.game.state is GameState.MENU and e.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                self.start_game()
        if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            st = self.game.state
            if st is GameState.MENU:
                for b in self.menu_buttons + [self.btn_start, self.btn_input]:
                    if b.click(e.pos):
                        break
            elif st is GameState.PLAYING:
                for b in self.play_buttons:
                    if b.click(e.pos):
                        break
                for rect, rgb in self.swatches:
                    if rect.collidepoint(e.pos):
                        self.canvas.set_color(rgb)
                for rect, size in self.brush_rects:
                    if rect.collidepoint(e.pos):
                        self.canvas.set_brush(size)
            elif st is GameState.RESULTS:
                for b in self.result_buttons:
                    if b.click(e.pos):
                        break
        return True

    # ------------------------------------------------------------ drawing the screens
    def text(self, s, font, color, pos, center=False):
        surf = font.render(s, True, color)
        rect = surf.get_rect(center=pos) if center else surf.get_rect(topleft=pos)
        self.screen.blit(surf, rect)
        return rect

    def header(self):
        self.text("AirSketch AI", self.f_h, ACCENT, (20, 14))
        g = self.game
        stats = f"Current {g.current_score:.0f}    Best {g.best_score:.0f}    Rounds {g.rounds_completed}"
        self.text(stats, self.f_b, INK, (W - 20 - self.f_b.size(stats)[0], 20))

    def draw_menu(self, mouse):
        self.text("AirSketch AI", self.f_title, ACCENT, (W // 2, 90), True)
        self.text("Draw in the air. Challenge AI. Beat your best score.", self.f_h, INK, (W // 2, 155), True)
        rules = [
            "1. Pick a difficulty and press Start. You'll see a reference picture.",
            "2. You have 60 seconds to redraw it by pointing your index finger at the webcam.",
            "3. Point = draw  |  Two fingers = lift the pen  |  Open palm (hold 1s) = clear.",
            "4. Submit (Enter) and the AI scores your drawing. Space toggles the pen, S skips.",
            "No webcam? Switch to mouse mode and drag with the left button.",
        ]
        y = 215
        for r in rules:
            self.text(r, self.f_b, INK, (W // 2, y), True)
            y += 36
        self.text("Difficulty", self.f_h, ACCENT2, (W // 2, 430), True)
        for b in self.menu_buttons:
            b.selected = b.label.lower() == self.game.difficulty
            b.draw(self.screen, self.f_b, mouse)
        self.btn_start.draw(self.screen, self.f_h, mouse)
        self.btn_input.label = f"Input: {'Webcam' if self.input_mode == 'webcam' else 'Mouse'} (click to switch)"
        self.btn_input.draw(self.screen, self.f_s, mouse)
        if self.camera_msg:
            for i, line in enumerate(wrap(self.camera_msg, self.f_s, 900)):
                self.text(line, self.f_s, BAD, (W // 2, 715 + i * 20), True)
        if self.game.history:
            self.header()

    def draw_preview(self, x, y):
        pygame.draw.rect(self.screen, PANEL, (x - 6, y - 6, PREVIEW[0] + 12, PREVIEW[1] + 12), border_radius=10)
        if self.input_mode == "mouse" or not self.camera:
            self.text("Mouse mode", self.f_b, MUTED, (x + PREVIEW[0] // 2, y + PREVIEW[1] // 2), True)
            return
        frame, _ = self.camera.latest()
        if frame is None:
            self.text("Starting camera...", self.f_b, MUTED, (x + PREVIEW[0] // 2, y + PREVIEW[1] // 2), True)
            return
        import cv2
        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), PREVIEW)
        self.screen.blit(pygame.image.frombytes(small.tobytes(), PREVIEW, "RGB"), (x, y))
        if self.hand.tip:
            px, py = int(x + self.hand.tip[0] * PREVIEW[0]), int(y + self.hand.tip[1] * PREVIEW[1])
            col = {POINT: GOOD, PEACE: ACCENT, OPEN: ACCENT2}.get(self.hand.gesture, BAD)
            pygame.draw.circle(self.screen, col, (px, py), 10, 3)
            pygame.draw.circle(self.screen, col, (px, py), 3)
        self.text(f"Gesture: {self.hand.gesture}", self.f_s, INK, (x, y + PREVIEW[1] + 8))

    def draw_play(self, mouse):
        g = self.game
        self.header()
        # reference
        self.text("Recreate this:", self.f_b, ACCENT2, (20, 70))
        pygame.draw.rect(self.screen, (255, 255, 255), (16, 106, 308, 308), border_radius=10)
        self.screen.blit(self.reference_surf, (20, 110))
        self.text(f"Difficulty: {g.challenge.difficulty}", self.f_s, MUTED, (20, 420))
        # timer
        tl = g.time_left
        col = BAD if tl <= 10 else (ACCENT if tl <= 20 else INK)
        self.text(f"{math.ceil(tl):02d}", self.f_timer, col, (20, 445))
        self.text("seconds left", self.f_s, MUTED, (130, 485))
        # canvas
        cx, cy = CANVAS_POS
        pygame.draw.rect(self.screen, ACCENT2, (cx - 4, cy - 4, CANVAS_W + 8, CANVAS_H + 8), border_radius=8)
        self.screen.blit(pygame.image.frombytes(self.canvas.array.tobytes(), (CANVAS_W, CANVAS_H), "RGB"), (cx, cy))
        cursor = None
        if self.input_mode == "webcam" and self.hand.tip and g.can_draw:
            cursor = normalized_to_canvas(*self.hand.tip)
            cursor = (cx + cursor[0], cy + cursor[1])
        elif self.input_mode == "mouse" and pygame.Rect(cx, cy, CANVAS_W, CANVAS_H).collidepoint(mouse):
            cursor = mouse
        if cursor:
            drawing = self.pen_enabled and (self.input_mode == "mouse" or self.hand.gesture == POINT)
            pygame.draw.circle(self.screen, GOOD if drawing else BAD, cursor, 11, 3)
        prog = self.clear_hold.progress(time.monotonic())
        if prog > 0:
            pygame.draw.rect(self.screen, ACCENT, (cx, cy + CANVAS_H + 10, int(CANVAS_W * prog), 8))
            self.text("Hold open palm to clear...", self.f_s, ACCENT, (cx, cy + CANVAS_H + 22))
        # right column
        x0 = 1020
        self.text("Webcam", self.f_b, ACCENT2, (x0, 70))
        self.draw_preview(x0, 110)
        self.text("Color", self.f_b, ACCENT2, (x0, 370))
        for rect, rgb in self.swatches:
            pygame.draw.rect(self.screen, rgb, rect, border_radius=8)
            if rgb == self.canvas.color:
                pygame.draw.rect(self.screen, INK, rect, 3, border_radius=8)
        self.text("Brush", self.f_b, ACCENT2, (x0, 450))
        for rect, size in self.brush_rects:
            pygame.draw.rect(self.screen, PANEL, rect, border_radius=8)
            pygame.draw.circle(self.screen, INK, rect.center, max(2, size // 2 + 1))
            if size == self.canvas.brush:
                pygame.draw.rect(self.screen, INK, rect, 3, border_radius=8)
        # buttons
        self.btn_pen.label = f"Pen: {'ON' if self.pen_enabled else 'OFF'}  (Space)"
        for b in self.play_buttons:
            b.draw(self.screen, self.f_b, mouse)

    def draw_evaluating(self):
        self.header()
        t = time.monotonic()
        self.text("The AI is judging your masterpiece...", self.f_h, ACCENT, (W // 2, H // 2 - 60), True)
        for i in range(8):
            a = t * 4 + i * math.pi / 4
            r = 6 + 3 * math.sin(t * 6 + i)
            pygame.draw.circle(self.screen, ACCENT2, (W // 2 + 50 * math.cos(a), H // 2 + 20 + 50 * math.sin(a)), r)

    def draw_results(self, mouse):
        self.header()
        r = self.result
        if r is None:
            return
        if not r.ok:
            self.text("Couldn't evaluate that round", self.f_h, BAD, (W // 2, 200), True)
            for i, line in enumerate(wrap(r.error, self.f_b, 800)):
                self.text(line, self.f_b, INK, (W // 2, 260 + i * 28), True)
        else:
            self.text(f"{r.score:.0f}", self.f_big, ACCENT, (W // 2, 118), True)
            label = grade_for_score(r.score) + ("  (approximate)" if r.approximate else "")
            self.text(label, self.f_h, INK, (W // 2, 190), True)
        # images
        ref = pygame.transform.smoothscale(self.reference_surf, (270, 270))
        y = 230
        pygame.draw.rect(self.screen, (255, 255, 255), (56, y - 4, 278, 278), border_radius=8)
        self.screen.blit(ref, (60, y))
        self.text("Reference", self.f_b, ACCENT2, (60, y + 282))
        dr = pygame.transform.smoothscale(self.final_surf, (360, 270))
        pygame.draw.rect(self.screen, (255, 255, 255), (356, y - 4, 368, 278), border_radius=8)
        self.screen.blit(dr, (360, y))
        self.text("Your drawing", self.f_b, ACCENT2, (360, y + 282))
        # text panel
        tx, ty = 770, 230
        if r.ok:
            self.text("AI sees:", self.f_b, ACCENT2, (tx, ty))
            self.text(r.recognized or "-", self.f_h, INK, (tx, ty + 24))
            ty += 70
            for line in wrap(r.feedback, self.f_b, 540):
                self.text(line, self.f_b, INK, (tx, ty))
                ty += 26
            ty += 8
            for s in r.suggestions:
                for i, line in enumerate(wrap(s, self.f_s, 520)):
                    self.text(("- " if i == 0 else "  ") + line, self.f_s, INK, (tx, ty))
                    ty += 22
            ty += 10
            for line in wrap(r.explanation, self.f_s, 540):
                self.text(line, self.f_s, MUTED, (tx, ty))
                ty += 20
            if r.note:
                for line in wrap(r.note, self.f_s, 540):
                    self.text(line, self.f_s, ACCENT, (tx, ty))
                    ty += 20
        for b in self.result_buttons:
            b.draw(self.screen, self.f_b, mouse)

    # ------------------------------------------------------------ main loop
    def run(self):
        running = True
        while running:
            mouse = pygame.mouse.get_pos()
            for e in pygame.event.get():
                running = self.handle_event(e) and running
            self.check_camera_health()
            self.poll_evaluation()
            if self.game.state is GameState.PLAYING:
                self.update_drawing()
            self.screen.fill(BG)
            st = self.game.state
            if st is GameState.MENU:
                self.draw_menu(mouse)
            elif st is GameState.PLAYING:
                self.draw_play(mouse)
            elif st is GameState.EVALUATING:
                self.draw_evaluating()
            else:
                self.draw_results(mouse)
            if time.monotonic() < self.toast_until:
                surf = self.f_b.render(self.toast, True, (30, 30, 50))
                rect = surf.get_rect(center=(W // 2, H - 24)).inflate(30, 14)
                pygame.draw.rect(self.screen, ACCENT, rect, border_radius=10)
                self.screen.blit(surf, surf.get_rect(center=rect.center))
            pygame.display.flip()
            self.clock.tick(60)
        self.stop_camera()
        pygame.quit()


def main():
    App().run()


if __name__ == "__main__":
    main()
