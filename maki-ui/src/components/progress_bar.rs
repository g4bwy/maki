use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

const BAR_CHAR: &str = "━";
const UNFILLED_COLOR: Color = Color::DarkGray;

pub struct ProgressBarConfig<'a> {
    pub ratio: f64,
    pub style: Style,
    pub label: Option<&'a str>,
    pub label_style: Option<Style>,
    pub bar_width: u16,
}

pub fn render(frame: &mut Frame, area: Rect, config: &ProgressBarConfig<'_>) {
    if area.is_empty() {
        return;
    }

    let ratio = config.ratio.clamp(0.0, 1.0);
    let width = config.bar_width as usize;
    let filled = (ratio * width as f64).round() as usize;

    let mut spans = Vec::with_capacity(width);

    if let Some(label) = config.label {
        let style = config.label_style.unwrap_or_default();
        spans.push(Span::styled(label, style));
    }

    for i in 0..width {
        let style = if i < filled {
            config.style
        } else {
            Style::new().fg(UNFILLED_COLOR)
        };
        spans.push(Span::styled(BAR_CHAR, style));
    }

    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    use test_case::test_case;

    fn render_gauge(ratio: f64, width: u16) -> TestBackend {
        let backend = TestBackend::new(width, 1);
        let mut terminal = Terminal::new(backend.clone()).unwrap();
        terminal
            .draw(|f| {
                render(
                    f,
                    Rect::new(0, 0, width, 1),
                    &ProgressBarConfig {
                        ratio,
                        style: Style::default(),
                        label: None,
                        label_style: None,
                        bar_width: width,
                    },
                );
            })
            .unwrap();
        backend
    }

    #[test_case(0.0, 20 ; "ratio_zero")]
    #[test_case(0.5, 20 ; "ratio_half")]
    #[test_case(1.0, 20 ; "ratio_full")]
    fn render_does_not_panic(ratio: f64, width: u16) {
        let _ = render_gauge(ratio, width);
    }

    #[test_case(1.5 ; "ratio_over_one_clamped")]
    #[test_case(-0.5 ; "ratio_negative_clamped")]
    fn render_clamps_ratio(ratio: f64) {
        let _ = render_gauge(ratio, 20);
    }

    #[test]
    fn render_zero_width_does_not_panic() {
        let backend = TestBackend::new(0, 1);
        let mut terminal = Terminal::new(backend.clone()).unwrap();
        terminal
            .draw(|f| {
                render(
                    f,
                    Rect::new(0, 0, 0, 1),
                    &ProgressBarConfig {
                        ratio: 0.5,
                        style: Style::default(),
                        label: None,
                        label_style: None,
                        bar_width: 0,
                    },
                );
            })
            .unwrap();
    }

    #[test]
    fn render_with_label_does_not_panic() {
        let backend = TestBackend::new(30, 1);
        let mut terminal = Terminal::new(backend.clone()).unwrap();
        terminal
            .draw(|f| {
                render(
                    f,
                    Rect::new(0, 0, 30, 1),
                    &ProgressBarConfig {
                        ratio: 0.5,
                        style: Style::default(),
                        label: Some(" PP:"),
                        label_style: None,
                        bar_width: 30,
                    },
                );
            })
            .unwrap();
    }
}
