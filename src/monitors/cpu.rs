use std::fs;
use std::sync::{LazyLock, Mutex};

use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use ratatui::Frame;

use crate::monitors::history::History;
use crate::theme::Theme;
use crate::widgets::meter;

#[derive(Clone, Default)]
pub struct CpuSnapshot {
    pub usage: f32,
    pub user_pct: f32,
    pub sys_pct: f32,
    pub iowait_pct: f32,
    pub cores: Vec<f32>,
    pub load: (f64, f64, f64),
    pub freq_mhz: u64,
    /// Per-sensor temps from hwmon (package first when the driver reports one).
    pub temps: Vec<f64>,
    /// Temperature of the physical core each logical CPU runs on, aligned
    /// with `cores`. `None` when the sensor or topology is unknown.
    pub thread_temps: Vec<Option<f64>>,
    /// One entry per physical CPU package (socket), sorted by socket id.
    /// Empty when no CPU hwmon driver is loaded.
    pub sockets: Vec<SocketTemp>,
}

/// Temperatures of one physical CPU package.
#[derive(Clone, Debug, PartialEq)]
pub struct SocketTemp {
    /// 0-based socket id (`physical_package_id`).
    pub socket: u32,
    /// The package-wide reading (Tdie / Package id N / Tctl).
    pub package_c: f64,
    /// Every sensor of this package, sorted by sensor index.
    pub sensors_c: Vec<f64>,
}

impl CpuSnapshot {
    /// Hottest sensor, or None when no sensor is available.
    pub fn max_temp(&self) -> Option<f64> {
        self.temps.iter().copied().reduce(f64::max)
    }

    /// Package temperature per socket, but only on multi-socket machines,
    /// where a single max would hide which CPU is hot.
    pub fn socket_temps(&self) -> Option<Vec<f64>> {
        (self.sockets.len() > 1).then(|| self.sockets.iter().map(|s| s.package_c).collect())
    }
}

static SNAP: LazyLock<Mutex<std::sync::Arc<CpuSnapshot>>> =
    LazyLock::new(|| Mutex::new(std::sync::Arc::new(CpuSnapshot::default())));
static HISTORY_USAGE: LazyLock<Mutex<History<240>>> = LazyLock::new(|| Mutex::new(History::new()));
static HISTORY_USER: LazyLock<Mutex<History<240>>> = LazyLock::new(|| Mutex::new(History::new()));
static HISTORY_SYS: LazyLock<Mutex<History<240>>> = LazyLock::new(|| Mutex::new(History::new()));
static HISTORY_IOWAIT: LazyLock<Mutex<History<240>>> = LazyLock::new(|| Mutex::new(History::new()));

pub fn snapshot() -> std::sync::Arc<CpuSnapshot> {
    SNAP.lock().unwrap().clone()
}

/// Called once per tick with a freshly refreshed sysinfo handle.
static PREV_STAT: Mutex<Option<(f64, f64, f64, f64)>> = Mutex::new(None); // (user, sys, iowait, total)

pub fn sample(sys: &sysinfo::System) {
    let mut user_pct = 0.0;
    let mut sys_pct = 0.0;
    let mut iowait_pct = 0.0;

    if let Ok(stat) = fs::read_to_string("/proc/stat") {
        if let Some(line) = stat.lines().find(|l| l.starts_with("cpu ")) {
            let p: Vec<f64> = line
                .split_whitespace()
                .skip(1)
                .filter_map(|s| s.parse().ok())
                .collect();
            if p.len() >= 8 {
                let user = p[0] + p[1];
                let sys_val = p[2] + p[5] + p[6];
                let iowait = p[4];
                let total = p.iter().sum::<f64>();

                let mut prev = PREV_STAT.lock().unwrap();
                if let Some((p_user, p_sys, p_iowait, p_total)) = *prev {
                    let d_total = total - p_total;
                    if d_total > 0.0 {
                        user_pct = ((user - p_user) / d_total * 100.0) as f32;
                        sys_pct = ((sys_val - p_sys) / d_total * 100.0) as f32;
                        iowait_pct = ((iowait - p_iowait) / d_total * 100.0) as f32;
                    }
                }
                *prev = Some((user, sys_val, iowait, total));
            }
        }
    }

    let la = sysinfo::System::load_average();
    let CoreTemps {
        all: temps,
        by_core,
        sockets,
    } = read_core_temps();
    let thread_temps = sys
        .cpus()
        .iter()
        .enumerate()
        .map(|(i, _)| {
            THREAD_CORE
                .get(i)
                .copied()
                .flatten()
                .and_then(|c| by_core.get(&c).copied())
        })
        .collect();
    let snap = CpuSnapshot {
        usage: sys.global_cpu_usage(),
        user_pct,
        sys_pct,
        iowait_pct,
        cores: sys.cpus().iter().map(|c| c.cpu_usage()).collect(),
        load: (la.one, la.five, la.fifteen),
        freq_mhz: sys.cpus().iter().map(|c| c.frequency()).max().unwrap_or(0),
        temps,
        thread_temps,
        sockets,
    };
    HISTORY_USAGE.lock().unwrap().push(snap.usage as f64);
    HISTORY_USER.lock().unwrap().push(snap.user_pct as f64);
    HISTORY_SYS.lock().unwrap().push(snap.sys_pct as f64);
    HISTORY_IOWAIT.lock().unwrap().push(snap.iowait_pct as f64);
    *SNAP.lock().unwrap() = std::sync::Arc::new(snap);
}

/// (socket, physical core id) of each logical CPU, from
/// `cpuN/topology/{physical_package_id,core_id}`, read once. Core ids repeat
/// on every socket, so both are needed to find a thread's sensor.
static THREAD_CORE: LazyLock<Vec<Option<(u32, u32)>>> = LazyLock::new(|| {
    let read = |dir: &str, f: &str| -> Option<u32> {
        fs::read_to_string(format!("{}/topology/{}", dir, f))
            .ok()?
            .trim()
            .parse()
            .ok()
    };
    (0..)
        .map_while(|i| {
            let dir = format!("/sys/devices/system/cpu/cpu{}", i);
            std::path::Path::new(&dir).exists().then(|| {
                let core = read(&dir, "core_id")?;
                Some((read(&dir, "physical_package_id").unwrap_or(0), core))
            })
        })
        .collect()
});

/// The CPU's own critical temperature (Tjmax, where throttling starts),
/// from `temp1_crit` / `temp1_max` of the CPU hwmon. 100° when unreported.
pub static TEMP_CRIT: LazyLock<f64> = LazyLock::new(|| {
    fs::read_dir("/sys/class/hwmon/")
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            fs::read_to_string(p.join("name"))
                .is_ok_and(|n| matches!(n.trim(), "coretemp" | "k10temp" | "zenpower"))
        })
        .find_map(|p| {
            ["temp1_crit", "temp1_max"].iter().find_map(|f| {
                fs::read_to_string(p.join(f))
                    .ok()?
                    .trim()
                    .parse::<f64>()
                    .ok()
                    .map(|v| v / 1000.0)
                    .filter(|c| (60.0..=130.0).contains(c))
            })
        })
        .unwrap_or(100.0)
});

/// "Core 3" -> 3. Package/Tctl labels are not per-core.
fn core_label_id(label: &str) -> Option<u32> {
    label.trim().strip_prefix("Core ")?.parse().ok()
}

/// "Package id 1" -> 1 (Intel coretemp names its package sensor this way).
fn package_label_id(label: &str) -> Option<u32> {
    label.trim().strip_prefix("Package id ")?.parse().ok()
}

/// The reading that stands for the whole package: Tdie, then the Intel
/// package sensor, then Tctl (which some AMD parts offset upwards), else the
/// first sensor.
fn package_reading(sensors: &[(String, f64)]) -> Option<f64> {
    let find = |pred: &dyn Fn(&str) -> bool| {
        sensors
            .iter()
            .find(|(l, _)| pred(l.trim()))
            .map(|(_, t)| *t)
    };
    find(&|l| l == "Tdie")
        .or_else(|| find(&|l| l.starts_with("Package id ")))
        .or_else(|| find(&|l| l == "Tctl"))
        .or_else(|| sensors.first().map(|(_, t)| *t))
}

/// One CPU hwmon device's sensors, as (label, °C) sorted by sensor index.
fn read_hwmon_sensors(dir: &std::path::Path) -> Vec<(String, f64)> {
    let mut sensors: Vec<(usize, String, f64)> = fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|te| {
            let fname = te.file_name().to_string_lossy().to_string();
            let num = fname
                .strip_prefix("temp")?
                .strip_suffix("_input")?
                .parse::<usize>()
                .ok()?;
            let v = fs::read_to_string(te.path())
                .ok()?
                .trim()
                .parse::<f64>()
                .ok()?;
            let label = fs::read_to_string(dir.join(format!("temp{}_label", num)))
                .map(|l| l.trim().to_string())
                .unwrap_or_default();
            Some((num, label, v / 1000.0))
        })
        .collect();
    sensors.sort_by_key(|(num, _, _)| *num);
    sensors.into_iter().map(|(_, l, t)| (l, t)).collect()
}

/// All CPU sensor temps (socket by socket, each sorted by sensor index), a
/// map from (socket, physical core id) to its temperature, and one package
/// reading per socket.
///
/// Each CPU package has its own hwmon device (`coretemp.N` on Intel, one
/// k10temp per data-fabric node on AMD). Sockets are numbered by the Intel
/// "Package id" label when present, else by the device's sysfs path order,
/// which follows the node number on AMD.
fn read_core_temps() -> CoreTemps {
    let mut out = CoreTemps::default();

    let mut hwmons: Vec<(std::path::PathBuf, std::path::PathBuf)> =
        fs::read_dir("/sys/class/hwmon/")
            .into_iter()
            .flatten()
            .flatten()
            .map(|e| e.path())
            .filter(|p| {
                fs::read_to_string(p.join("name"))
                    .is_ok_and(|n| matches!(n.trim(), "coretemp" | "k10temp" | "zenpower"))
            })
            .map(|p| {
                (
                    fs::canonicalize(p.join("device")).unwrap_or_else(|_| p.clone()),
                    p,
                )
            })
            .collect();
    hwmons.sort();

    for (idx, (_, dir)) in hwmons.iter().enumerate() {
        let sensors = read_hwmon_sensors(dir);
        if sensors.is_empty() {
            continue;
        }
        let socket = sensors
            .iter()
            .find_map(|(l, _)| package_label_id(l))
            .unwrap_or(idx as u32);
        for (label, t) in &sensors {
            if let Some(core) = core_label_id(label) {
                out.by_core.insert((socket, core), *t);
            }
        }
        if let Some(package) = package_reading(&sensors) {
            out.sockets.push(SocketTemp {
                socket,
                package_c: package,
                sensors_c: sensors.iter().map(|(_, t)| *t).collect(),
            });
        }
        out.all.extend(sensors.iter().map(|(_, t)| *t));
    }
    out.sockets.sort_by_key(|s| s.socket);

    if out.all.is_empty() {
        let comps = sysinfo::Components::new_with_refreshed_list();
        out.all = comps
            .iter()
            .filter_map(|c| c.temperature().map(f64::from))
            .collect();
    }
    out
}

#[derive(Default)]
struct CoreTemps {
    all: Vec<f64>,
    by_core: std::collections::HashMap<(u32, u32), f64>,
    sockets: Vec<SocketTemp>,
}

/// One row of per-core load: each core gets an equal slice filled with a
/// block whose height and colour track its usage, e.g. `▂▂ ▇▇ ▁▁ ▅▅`.
fn render_core_strip(f: &mut Frame, area: Rect, theme: &Theme, cores: &[f32]) {
    const BARS: [char; 8] = [' ', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    let n = cores.len();
    let w = area.width as usize;
    if n == 0 || w < n {
        return;
    }
    // Leave a one-cell gap between cores when there's room for it.
    let slot = w / n;
    let (cell, gap) = if slot >= 3 { (slot - 1, 1) } else { (slot, 0) };
    let total_w = n * cell + n.saturating_sub(1) * gap;
    let pad = w.saturating_sub(total_w) / 2;
    let mut spans = Vec::with_capacity(n * 2 + 1);
    if pad > 0 {
        spans.push(Span::raw(" ".repeat(pad)));
    }
    for (i, &u) in cores.iter().enumerate() {
        if i > 0 && gap > 0 {
            spans.push(Span::raw(" ".repeat(gap)));
        }
        let idx = ((u / 100.0) * 7.0).round().clamp(0.0, 7.0) as usize;
        spans.push(Span::styled(
            BARS[idx].to_string().repeat(cell),
            Style::default().fg(theme.usage(u as f64)),
        ));
    }
    f.render_widget(Paragraph::new(Line::from(spans)), area);
}

/// Width of the `c{i}` label column, one cell wider than the longest label.
fn core_label_width(core_count: usize) -> usize {
    let digits = core_count.saturating_sub(1).max(1).ilog10() as usize + 1;
    digits + 2
}

/// Columns in the per-core meter grid. Starts at two (one when narrow) and
/// adds columns until every core fits in `max_rows`, as long as each cell
/// still has room for the label, a short bar and the percentage.
fn cores_per_line(core_count: usize, width: u16, max_rows: u16, label_w: usize) -> usize {
    const SPACING: usize = 2;
    let base = if width >= 36 { 2 } else { 1 };
    let min_cell = label_w + 3 + 5;
    let max_cols = ((width as usize + SPACING) / (min_cell + SPACING)).max(1);
    let needed = core_count.div_ceil((max_rows as usize).max(1));
    needed.max(base).min(max_cols.max(base))
}

/// Monitor-page CPU panel: header, big history graph, per-core meters.
pub fn render(f: &mut Frame, area: Rect, theme: &Theme, is_detailed: bool) {
    if area.height < 3 || area.width < 20 {
        return;
    }
    let snap = snapshot();
    let core_count = snap.cores.len();
    let label_w = core_label_width(core_count);

    // Cap the core block so the graph always keeps at least 2 rows.
    let max_rows = area.height.saturating_sub(4);
    let per_line = cores_per_line(core_count, area.width, max_rows, label_w);
    let mut core_rows = if is_detailed {
        core_count.div_ceil(per_line) as u16
    } else {
        0
    };
    core_rows = core_rows.min(max_rows);

    // Compact mode: a one-row per-core heat strip under the graph.
    let strip = !is_detailed && area.height >= 6 && core_count > 0;
    let mut constraints = vec![Constraint::Length(1), Constraint::Min(2)];
    if core_rows > 0 {
        constraints.push(Constraint::Length(1));
        constraints.push(Constraint::Length(core_rows));
    } else if strip {
        constraints.push(Constraint::Length(1));
    }
    let chunks = Layout::vertical(constraints).split(area);

    let color = theme.usage(snap.usage as f64);
    // In compact mode the panel title already carries usage/temp/freq.
    let mut header = if is_detailed {
        vec![Span::styled(
            format!("{:>3.0}%  ", snap.usage),
            Style::default().fg(color),
        )]
    } else {
        Vec::new()
    };

    if snap.user_pct > 0.5 || snap.sys_pct > 0.5 || snap.iowait_pct > 0.5 {
        header.push(Span::styled(
            format!("usr {:.0}%", snap.user_pct),
            Style::default().fg(theme.secondary),
        ));
        header.push(Span::styled(
            format!("  sys {:.0}%", snap.sys_pct),
            Style::default().fg(theme.yellow),
        ));
        header.push(Span::styled(
            format!("  io {:.0}%", snap.iowait_pct),
            Style::default().fg(theme.red),
        ));
    }

    // Most to least important, so trimming from the end drops the least
    // useful facts first instead of clipping a number at the border.
    // Multi-socket: one reading per CPU, numbered from 1 like the physical
    // sockets on the board ("cpu1 38°C  cpu2 39°C"); otherwise the hottest.
    let temps: Vec<(String, f64)> = match (is_detailed, snap.socket_temps()) {
        (false, _) => Vec::new(),
        (true, Some(per_socket)) => per_socket
            .iter()
            .enumerate()
            .map(|(i, t)| (format!("  cpu{} {:.0}°C", i + 1, t), *t))
            .collect(),
        (true, None) => snap
            .max_temp()
            .map(|t| (format!("  {:.0}°C", t), t))
            .into_iter()
            .collect(),
    };
    for (text, t) in temps {
        header.push(Span::styled(text, Style::default().fg(theme.temp(t))));
    }
    if snap.freq_mhz > 0 && is_detailed {
        header.push(Span::styled(
            format!("  {:.1}GHz", snap.freq_mhz as f64 / 1000.0),
            Style::default().fg(theme.text),
        ));
    }
    header.push(Span::styled(
        format!(
            "  load {:.2} {:.2} {:.2}",
            snap.load.0, snap.load.1, snap.load.2
        ),
        Style::default().fg(theme.dim),
    ));
    if is_detailed {
        header.push(Span::styled(
            format!("  {} threads", core_count),
            Style::default().fg(theme.dim),
        ));
    }
    let mut header = Line::from(header);
    while header.width() > area.width as usize && header.spans.len() > 1 {
        header.spans.pop();
    }
    f.render_widget(Paragraph::new(header), chunks[0]);

    use crate::widgets::block_graph::BlockGraph;

    let _points = chunks[1].width as usize; // blockgraph handles its own sub_w
    let hist_usage = HISTORY_USAGE.lock().unwrap().recent(1000);

    // We force braille style for the mirrored CPU graph by temporarily setting it,
    // or wait, blockgraph respects the global setting.
    // Let's assume the user has braille style selected, or block style works too.
    f.render_widget(
        BlockGraph::new(&hist_usage)
            .pending(theme.dim)
            .max(100.0)
            .mirrored(true)
            .colors(theme.accent, theme.yellow, theme.red),
        chunks[1],
    );
    if strip && core_rows == 0 {
        render_core_strip(f, chunks[2], theme, &snap.cores);
        return;
    }
    if core_rows == 0 {
        return;
    }
    let cols = Layout::horizontal(vec![Constraint::Ratio(1, per_line as u32); per_line])
        .spacing(2)
        .split(chunks[3]);
    // Fixed columns so every bar starts and ends at the same x: the temp
    // column is reserved whenever any core has a sensor, and dropped when the
    // cell is too narrow to fit a useful bar alongside it.
    let any_temp = snap.thread_temps.iter().any(Option::is_some);
    let cell_w = cols[0].width as usize;
    let show_temp = any_temp && cell_w >= label_w + 6 + 5 + 4;
    let right_w = 5 + if show_temp { 4 } else { 0 };
    for (i, &usage) in snap.cores.iter().enumerate() {
        let row = (i / per_line) as u16;
        if row >= core_rows {
            break;
        }
        let col = cols[i % per_line];
        let c = theme.usage(usage as f64);
        let bar_w = (col.width as usize).saturating_sub(label_w + right_w);
        let (on, off) = meter::track(usage as f64 / 100.0, bar_w);
        let mut spans = vec![
            Span::styled(
                format!("{:<w$}", format!("c{}", i), w = label_w),
                Style::default().fg(theme.dim),
            ),
            Span::styled(on, Style::default().fg(c)),
            Span::styled(off, Style::default().fg(theme.surface)),
            Span::styled(format!("{:>4.0}%", usage), Style::default().fg(c)),
        ];
        if show_temp {
            spans.push(match snap.thread_temps.get(i).copied().flatten() {
                Some(t) => Span::styled(format!("{:>3.0}°", t), Style::default().fg(theme.temp(t))),
                None => Span::raw("    "),
            });
        }
        f.render_widget(
            Paragraph::new(Line::from(spans)),
            Rect::new(col.x, col.y + row, col.width, 1),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_core_labels_map_to_core_ids() {
        assert_eq!(core_label_id("Core 3\n"), Some(3));
        assert_eq!(core_label_id("Package id 0"), None);
        assert_eq!(core_label_id("Tctl"), None);
    }

    fn sensors(list: &[(&str, f64)]) -> Vec<(String, f64)> {
        list.iter().map(|(l, t)| (l.to_string(), *t)).collect()
    }

    #[test]
    fn package_labels_name_the_socket() {
        assert_eq!(package_label_id("Package id 1\n"), Some(1));
        assert_eq!(package_label_id("Core 1"), None);
        assert_eq!(package_label_id("Tctl"), None);
    }

    #[test]
    fn package_reading_prefers_die_then_package_then_tctl() {
        // AMD EPYC k10temp: Tctl plus one sensor per CCD.
        let amd = sensors(&[("Tctl", 38.6), ("Tccd1", 33.5), ("Tccd3", 34.5)]);
        assert_eq!(package_reading(&amd), Some(38.6));
        // Ryzen with an offset Tctl: Tdie is the real die temperature.
        let tdie = sensors(&[("Tctl", 60.0), ("Tdie", 40.0)]);
        assert_eq!(package_reading(&tdie), Some(40.0));
        // Intel coretemp: the package sensor, not the hottest core.
        let intel = sensors(&[("Package id 0", 45.0), ("Core 0", 50.0)]);
        assert_eq!(package_reading(&intel), Some(45.0));
        // Unlabelled sensors fall back to the first one.
        assert_eq!(
            package_reading(&sensors(&[("", 30.0), ("", 31.0)])),
            Some(30.0)
        );
        assert_eq!(package_reading(&[]), None);
    }

    #[test]
    fn socket_temps_only_split_on_multi_socket_machines() {
        let socket = |socket, package_c| SocketTemp {
            socket,
            package_c,
            sensors_c: vec![package_c],
        };
        let one = CpuSnapshot {
            sockets: vec![socket(0, 40.0)],
            ..Default::default()
        };
        assert_eq!(one.socket_temps(), None);
        let two = CpuSnapshot {
            sockets: vec![socket(0, 38.6), socket(1, 39.4)],
            ..Default::default()
        };
        assert_eq!(two.socket_temps(), Some(vec![38.6, 39.4]));
    }

    #[test]
    fn label_width_fits_longest_core_label() {
        assert_eq!(core_label_width(8), 3); // "c7 "
        assert_eq!(core_label_width(16), 4); // "c15 "
        assert_eq!(core_label_width(128), 5); // "c127 "
    }

    #[test]
    fn grid_keeps_two_columns_when_cores_fit() {
        assert_eq!(cores_per_line(16, 80, 20, 4), 2);
        assert_eq!(cores_per_line(4, 30, 20, 3), 1);
    }

    #[test]
    fn grid_widens_so_128_threads_fit() {
        let label_w = core_label_width(128);
        // 200x50 zoomed panel: 42 rows free -> 4 columns, 32 rows.
        let cols = cores_per_line(128, 196, 42, label_w);
        assert_eq!(cols, 4);
        assert!(128usize.div_ceil(cols) <= 42);
        // Width caps the columns: 13-cell minimum + 2 spacing in 100 cols.
        assert_eq!(cores_per_line(128, 100, 10, label_w), 6);
    }

    #[test]
    fn detailed_panel_draws_every_one_of_128_threads() {
        use ratatui::{backend::TestBackend, Terminal};
        *SNAP.lock().unwrap() = std::sync::Arc::new(CpuSnapshot {
            cores: (0..128).map(|i| (i % 100) as f32).collect(),
            thread_temps: vec![None; 128],
            temps: vec![38.6, 33.5, 39.4, 32.5],
            sockets: vec![
                SocketTemp {
                    socket: 0,
                    package_c: 38.6,
                    sensors_c: vec![38.6, 33.5],
                },
                SocketTemp {
                    socket: 1,
                    package_c: 39.4,
                    sensors_c: vec![39.4, 32.5],
                },
            ],
            ..Default::default()
        });
        let theme = Theme::dark();
        let mut term = Terminal::new(TestBackend::new(196, 46)).unwrap();
        term.draw(|f| render(f, f.area(), &theme, true)).unwrap();
        let buf = term.backend().buffer();
        let screen: String = (0..buf.area.height)
            .map(|y| {
                (0..buf.area.width)
                    .map(|x| buf[(x, y)].symbol())
                    .collect::<String>()
                    + "\n"
            })
            .collect();
        for i in [0, 63, 64, 127] {
            assert!(
                screen.contains(&format!("c{i} ")),
                "c{i} missing:\n{screen}"
            );
        }
        let header = screen.lines().next().unwrap();
        assert!(
            header.contains("cpu1 39°C") && header.contains("cpu2 39°C"),
            "{header}"
        );
    }
}
