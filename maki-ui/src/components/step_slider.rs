use crate::theme::current;

use crossterm::event::KeyEvent;
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use unicode_width::UnicodeWidthStr;

const TRACK_CHAR: &str = "━";
const CIRCLE_EMPTY: &str = "○";
const CIRCLE_FILLED: &str = "◉";
const ARROW_LEFT: &str = "◀";
const ARROW_RIGHT: &str = "▶";

pub struct Step {
    pub label: &'static str,
    pub color: Color,
    pub desc: &'static str,
}

#[derive(Debug, PartialEq, Eq)]
pub enum StepSliderAction {
    Consumed,
    Changed(usize),
}

pub struct StepSlider {
    selected: usize,
    visible: bool,
}

#[allow(dead_code)]
impl StepSlider {
    pub fn new() -> Self {
        Self {
            selected: 0,
            visible: false,
        }
    }

    pub fn with_selected(index: usize) -> Self {
        Self {
            selected: index,
            visible: false,
        }
    }

    pub fn selected(&self) -> usize {
        self.selected
    }

    pub fn set_selected(&mut self, index: usize) {
        self.selected = index;
    }

    pub fn is_visible(&self) -> bool {
        self.visible
    }

    pub fn set_visible(&mut self, visible: bool) {
        self.visible = visible;
    }

    pub fn show(&mut self) {
        self.visible = true;
    }

    pub fn hide(&mut self) {
        self.visible = false;
    }

    pub fn prev(&mut self) -> StepSliderAction {
        if self.selected > 0 {
            self.selected -= 1;
            StepSliderAction::Changed(self.selected)
        } else {
            StepSliderAction::Consumed
        }
    }

    pub fn next_bounded(&mut self, max: usize) -> StepSliderAction {
        if self.selected < max {
            self.selected += 1;
            StepSliderAction::Changed(self.selected)
        } else {
            StepSliderAction::Consumed
        }
    }

    pub fn handle_key(&mut self, key_event: KeyEvent, step_count: usize) -> StepSliderAction {
        use crossterm::event::KeyCode;
        match key_event.code {
            KeyCode::Left | KeyCode::Char('h') => self.prev(),
            KeyCode::Right | KeyCode::Char('l') => self.next_bounded(step_count - 1),
            KeyCode::Char(c) => {
                let idx = (c as u8).saturating_sub(b'1') as usize;
                if idx < step_count {
                    self.selected = idx;
                    StepSliderAction::Changed(idx)
                } else {
                    StepSliderAction::Consumed
                }
            }
            _ => StepSliderAction::Consumed,
        }
    }

    pub fn height() -> u16 {
        2
    }

    pub fn render(&self, frame: &mut Frame, area: Rect, label: &str, steps: &[Step]) {
        if !self.visible || area.is_empty() || steps.is_empty() || area.height < 2 {
            return;
        }

        let t = current();
        let color = steps[self.selected].color;
        let n = steps.len();

        // Track width: ~30% of available area, minus arrow padding
        let track_width = ((area.width as f64) * 0.3) as usize - 2;
        let total_width = label.width() + 2 + track_width + 2;
        if total_width > area.width as usize {
            return;
        }

        // Marker positions evenly distributed across track
        let marker_pos = Self::compute_positions(track_width, n);
        let selected_pos = marker_pos[self.selected];
        let marker_set: std::collections::HashSet<usize> = marker_pos.iter().copied().collect();

        let max_label = steps.iter().map(|s| s.label.width()).max().unwrap_or(0);
        let display_width = (total_width + max_label).min(area.width as usize);
        let para_area = Rect {
            x: area.x,
            y: area.y,
            width: display_width as u16,
            height: 2.min(area.height),
        };

        // Row 0: "Label  ◀━━━◉━━━○━━━▶"
        let mut row0 = Vec::new();
        row0.push(Span::styled(label.to_string(), Style::new().fg(t.foreground)));
        row0.push(Span::raw("  "));
        let left_col = if self.selected > 0 { color } else { t.foreground };
        row0.push(Span::styled(ARROW_LEFT, Style::new().fg(left_col)));
        row0.push(Span::raw(" "));

        let first_marker = marker_pos[0];
        let last_marker = marker_pos[n - 1];

        for i in 0..track_width {
            if marker_set.contains(&i) {
                let idx = marker_pos.iter().position(|&p| p == i).unwrap();
                let ch = if idx == self.selected { CIRCLE_FILLED } else { CIRCLE_EMPTY };
                row0.push(Span::styled(ch, Style::new().fg(color)));
            } else if i >= first_marker && i <= last_marker {
                let track_col = if i < selected_pos { color } else { Color::Gray };
                row0.push(Span::styled(TRACK_CHAR, Style::new().fg(track_col)));
            } else {
                row0.push(Span::raw(" "));
            }
        }

        row0.push(Span::raw(" "));
        let right_col = if self.selected < n - 1 { color } else { t.foreground };
        row0.push(Span::styled(ARROW_RIGHT, Style::new().fg(right_col)));

        // Row 1: align labels under markers (label + 2 spaces + arrow + space = offset)
        let label_pad = label.width() + 2 + 1 + 1;
        let max_label = steps.iter().map(|s| s.label.width()).max().unwrap_or(0);
        let row1_width = label_pad + track_width + max_label;
        let mut row1 = vec![Span::raw(" "); row1_width];

        for (i, step) in steps.iter().enumerate() {
            let pos = label_pad + marker_pos[i];
            let style = if i == self.selected {
                Style::new().fg(step.color)
            } else {
                Style::new().fg(Color::Gray)
            };
            for (j, ch) in step.label.chars().enumerate() {
                if pos + j < row1.len() {
                    row1[pos + j] = Span::styled(ch.to_string(), style);
                }
            }
        }

        frame.render_widget(
            Paragraph::new(vec![Line::from(row0), Line::from(row1)]),
            para_area,
        );
    }

    fn compute_positions(track_width: usize, count: usize) -> Vec<usize> {
        if count == 1 {
            return vec![track_width / 2];
        }
        // First marker at 0, last marker at track_width - 1, evenly spaced
        let gaps = count - 1;
        let span = track_width - 2; // Distance between first and last marker
        let step = span / gaps;
        let mut pos = Vec::with_capacity(count);
        for i in 0..count {
            if i == 0 {
                pos.push(0);
            } else if i == count - 1 {
                pos.push(track_width - 1);
            } else {
                pos.push(i * step);
            }
        }
        pos
    }

    pub fn description_line(&self, steps: &[Step]) -> Line<'static> {
        let t = current();
        let step = &steps[self.selected];
        Line::from(Span::styled(
            step.desc.to_string(),
            Style::new().fg(t.foreground).add_modifier(Modifier::ITALIC),
        ))
    }
}

impl Default for StepSlider {
    fn default() -> Self {
        Self::new()
    }
}

pub const EFFORT_STEPS: &[Step] = &[
    Step {
        label: "low",
        color: Color::Green,
        desc: "Faster responses, lighter reasoning \u{2013} great for simpler tasks",
    },
    Step {
        label: "medium",
        color: Color::Yellow,
        desc: "Balanced speed and reasoning quality for most tasks",
    },
    Step {
        label: "high",
        color: Color::Red,
        desc: "Deepest reasoning for complex problems \u{2013} slower but strongest",
    },
];

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyEventState};
    use test_case::test_case;

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent {
            code,
            modifiers: crossterm::event::KeyModifiers::NONE,
            kind: KeyEventKind::Press,
            state: KeyEventState::NONE,
        }
    }

    const STEPS: &[Step] = &[
        Step { label: "a", color: Color::Green, desc: "desc a" },
        Step { label: "b", color: Color::Yellow, desc: "desc b" },
        Step { label: "c", color: Color::Red, desc: "desc c" },
    ];

    #[test]
    fn default_state() {
        let s = StepSlider::new();
        assert_eq!(s.selected(), 0);
        assert!(!s.is_visible());
    }

    #[test]
    fn with_selected_sets_index() {
        let s = StepSlider::with_selected(2);
        assert_eq!(s.selected(), 2);
    }

    #[test_case(0, KeyCode::Left, 0, StepSliderAction::Consumed ; "left_at_start")]
    #[test_case(1, KeyCode::Left, 0, StepSliderAction::Changed(0) ; "left_from_mid")]
    #[test_case(2, KeyCode::Left, 1, StepSliderAction::Changed(1) ; "left_from_end")]
    #[test_case(0, KeyCode::Right, 1, StepSliderAction::Changed(1) ; "right_from_start")]
    #[test_case(1, KeyCode::Right, 2, StepSliderAction::Changed(2) ; "right_from_mid")]
    #[test_case(2, KeyCode::Right, 2, StepSliderAction::Consumed ; "right_at_end")]
    fn navigation(start: usize, code: KeyCode, expected_idx: usize, expected_action: StepSliderAction) {
        let mut s = StepSlider::with_selected(start);
        let action = s.handle_key(key(code), STEPS.len());
        assert_eq!(action, expected_action);
        assert_eq!(s.selected(), expected_idx);
    }

    #[test_case('1', 0 ; "digit_1")]
    #[test_case('2', 1 ; "digit_2")]
    #[test_case('3', 2 ; "digit_3")]
    #[test_case('4', 0 ; "digit_4_out_of_range")]
    fn digit_shortcut(ch: char, expected: usize) {
        let mut s = StepSlider::new();
        s.handle_key(key(KeyCode::Char(ch)), STEPS.len());
        assert_eq!(s.selected(), expected);
    }

    #[test]
    fn visibility_toggle() {
        let mut s = StepSlider::new();
        assert!(!s.is_visible());
        s.show();
        assert!(s.is_visible());
        s.hide();
        assert!(!s.is_visible());
    }

    #[test]
    fn set_selected_updates() {
        let mut s = StepSlider::new();
        s.set_selected(2);
        assert_eq!(s.selected(), 2);
    }

    #[test]
    fn height_is_constant() {
        assert_eq!(StepSlider::height(), 2);
    }

    #[test]
    fn description_line_returns_italic() {
        let s = StepSlider::new();
        let line = s.description_line(STEPS);
        assert!(!line.spans.is_empty());
    }

    #[test]
    fn effort_steps_have_three_levels() {
        assert_eq!(EFFORT_STEPS.len(), 3);
        assert_eq!(EFFORT_STEPS[0].label, "low");
        assert_eq!(EFFORT_STEPS[1].label, "medium");
        assert_eq!(EFFORT_STEPS[2].label, "high");
    }

    #[test]
    fn right_arrow_at_end_returns_consumed() {
        let mut s = StepSlider::with_selected(2);
        let action = s.handle_key(key(KeyCode::Right), 3);
        assert_eq!(action, StepSliderAction::Consumed);
        assert_eq!(s.selected(), 2);
    }

    #[test]
    fn left_arrow_at_start_returns_consumed() {
        let mut s = StepSlider::new();
        let action = s.handle_key(key(KeyCode::Left), 3);
        assert_eq!(action, StepSliderAction::Consumed);
        assert_eq!(s.selected(), 0);
    }
}
