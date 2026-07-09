use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::Style;
use ratatui::symbols;
use ratatui::widgets::LineGauge;

pub struct ProgressBarConfig<'a> {
    pub ratio: f64,
    pub style: Style,
    pub label: Option<&'a str>,
    pub use_unicode_weight: bool,
}

pub fn render(frame: &mut Frame, area: Rect, config: &ProgressBarConfig<'_>) {
    if area.is_empty() {
        return;
    }

    let ratio = config.ratio.clamp(0.0, 1.0);

    let mut gauge = LineGauge::default().ratio(ratio);

    if config.use_unicode_weight {
        gauge = gauge
            .filled_symbol(symbols::block::FULL)
            .unfilled_symbol(" ");
    } else {
        gauge = gauge
            .filled_symbol(symbols::line::THICK_HORIZONTAL)
            .unfilled_symbol(symbols::line::HORIZONTAL);
    }

    gauge = gauge.filled_style(config.style);

    if let Some(label) = config.label {
        gauge = gauge.label(label);
    }

    frame.render_widget(gauge, area);
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    use test_case::test_case;

    fn render_gauge(ratio: f64, width: u16, use_unicode: bool) -> TestBackend {
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
                        use_unicode_weight: use_unicode,
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
        let _ = render_gauge(ratio, width, false);
    }

    #[test_case(1.5 ; "ratio_over_one_clamped")]
    #[test_case(-0.5 ; "ratio_negative_clamped")]
    fn render_clamps_ratio(ratio: f64) {
        let _ = render_gauge(ratio, 20, false);
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
                        use_unicode_weight: false,
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
                        label: Some("test"),
                        use_unicode_weight: false,
                    },
                );
            })
            .unwrap();
    }

    #[test]
    fn render_unicode_mode_does_not_panic() {
        let _ = render_gauge(0.375, 20, true);
    }
}
