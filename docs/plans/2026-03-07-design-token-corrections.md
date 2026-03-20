# Phase 1: Design Token Corrections — Signal Discipline & Palette Unification

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Restore orange as a signal-only color by making resting borders neutral, unify the surface palette across both repos, and replace hardcoded hex values with design token references.

**Architecture:** Two theme.css files define the design system. Repo A (f1) is the canonical palette. Repo B (omnisense) drifts on background, card, secondary, and muted-foreground values. All component files use hardcoded hex/rgba instead of Tailwind token classes. This plan changes tokens at the source, then sweeps components file-by-file.

**Tech Stack:** Tailwind CSS v4, CSS custom properties, Recharts (chart grid strokes)

---

## Scope & Constraints

- **Token changes only.** No component logic, no layout changes, no prop changes.
- **Non-breaking.** Every change is a visual-only color swap. No DOM structure changes.
- **Two repos touched:**
  - Repo A: `/home/pedrad/javier_project_folder/f1/frontend/`
  - Repo B: `/home/pedrad/javier_project_folder/onmisense/Mclarenpredictivemaintenanceplatform/`

## Canonical Palette (Repo A values — the standard)

```
--background:       #0D1117
--card:             #1A1F2E
--secondary:        #222838
--muted:            #222838
--muted-foreground: #9090A8
--border:           rgba(255, 255, 255, 0.06)    ← CHANGED from orange
--border-strong:    rgba(255, 255, 255, 0.10)    ← CHANGED from orange
--border-accent:    rgba(255, 128, 0, 0.35)      ← KEPT (signal-only)
--sidebar:          #0D1117
--sidebar-border:   rgba(255, 255, 255, 0.06)    ← CHANGED from orange
--chart-grid:       rgba(255, 255, 255, 0.05)    ← CHANGED from orange
--chart-3:          #05DF72                       ← Repo A canonical
--chart-5:          #FB2C36                       ← Repo A canonical
--destructive:      #FB2C36                       ← Repo A canonical
```

---

### Task 1: Update Repo A theme.css — Neutral Borders & Grid

**Files:**
- Modify: `/home/pedrad/javier_project_folder/f1/frontend/src/styles/theme.css`

**Step 1: Change border tokens from orange to neutral white**

In `:root`, change:
```css
/* BEFORE */
--border: rgba(255, 128, 0, 0.12);
--border-strong: rgba(255, 128, 0, 0.20);

/* AFTER */
--border: rgba(255, 255, 255, 0.06);
--border-strong: rgba(255, 255, 255, 0.10);
```

Keep `--border-accent: rgba(255, 128, 0, 0.35);` unchanged — this is for signal states.

**Step 2: Change sidebar border to neutral**

```css
/* BEFORE */
--sidebar-border: rgba(255, 128, 0, 0.12);

/* AFTER */
--sidebar-border: rgba(255, 255, 255, 0.06);
```

**Step 3: Verify chart-grid is already neutral**

Confirm `--chart-grid: rgba(255, 255, 255, 0.06);` — this is already correct in Repo A.

**Step 4: Commit**

```bash
cd /home/pedrad/javier_project_folder/f1
git add frontend/src/styles/theme.css
git commit -m "fix(tokens): neutral borders — reserve orange for signals only"
```

---

### Task 2: Update Repo B theme.css — Unify to Canonical Palette

**Files:**
- Modify: `/home/pedrad/javier_project_folder/onmisense/Mclarenpredictivemaintenanceplatform/src/styles/theme.css`

**Step 1: Replace `:root` block with canonical values**

Change these values in `:root`:
```css
/* BEFORE → AFTER */
--background: #0a0a12;           → --background: #0D1117;
--foreground: #e8e8f0;           → --foreground: #E8E8F0;
--card: #12121e;                 → --card: #1A1F2E;
--card-foreground: #e8e8f0;     → --card-foreground: #E8E8F0;
--popover: #12121e;              → --popover: #1A1F2E;
--popover-foreground: #e8e8f0;  → --popover-foreground: #E8E8F0;
--primary-foreground: #0a0a12;  → --primary-foreground: #0D1117;
--secondary: #1a1a2e;           → --secondary: #222838;
--secondary-foreground: #e8e8f0;→ --secondary-foreground: #E8E8F0;
--muted: #1a1a2e;               → --muted: #222838;
--muted-foreground: #8888a0;    → --muted-foreground: #9090A8;
--accent-foreground: #0a0a12;   → --accent-foreground: #0D1117;
--destructive: #ef4444;          → --destructive: #FB2C36;
--border: rgba(255, 128, 0, 0.12); → --border: rgba(255, 255, 255, 0.06);
--input-background: #1a1a2e;    → --input-background: #222838;
--switch-background: #2a2a3e;   → --switch-background: #2A3142;
--chart-3: #22c55e;              → --chart-3: #05DF72;
--chart-5: #ef4444;              → --chart-5: #FB2C36;
--sidebar: #0d0d18;              → --sidebar: #0D1117;
--sidebar-primary-foreground: #0a0a12; → --sidebar-primary-foreground: #0D1117;
--sidebar-accent: #1a1a2e;      → --sidebar-accent: #222838;
--sidebar-accent-foreground: #e8e8f0; → --sidebar-accent-foreground: #E8E8F0;
--sidebar-border: rgba(255, 128, 0, 0.12); → --sidebar-border: rgba(255, 255, 255, 0.06);
```

**Step 2: Apply same changes to `.dark` block**

The `.dark` block duplicates `:root`. Apply the identical substitutions.

**Step 3: Add missing tokens from Repo A**

Add after `--switch-background`:
```css
/* Borders — 3-tier system */
--border-strong: rgba(255, 255, 255, 0.10);
--border-accent: rgba(255, 128, 0, 0.35);

/* Text tiers */
--text-secondary: #C0C0D0;
--text-muted: #9090A8;
--text-faint: #6E6E88;

/* Semantic status colors */
--success: #05DF72;
--success-soft: rgba(5, 223, 114, 0.10);
--warning: #F59E0B;
--warning-soft: rgba(245, 158, 11, 0.10);
--danger: #FB2C36;
--danger-soft: rgba(251, 44, 54, 0.10);
--info: #3B82F6;
--info-soft: rgba(59, 130, 246, 0.10);

/* Accent soft/muted */
--accent-soft: rgba(255, 128, 0, 0.10);
--accent-muted: rgba(255, 128, 0, 0.18);

/* Shadows — 3-tier */
--shadow-sm: 0 1px 3px rgba(0, 0, 0, 0.3);
--shadow-md: 0 4px 12px rgba(0, 0, 0, 0.4);
--shadow-lg: 0 8px 24px rgba(0, 0, 0, 0.5);

/* Charts */
--chart-grid: rgba(255, 255, 255, 0.05);
--chart-axis: #9090A8;

/* Surface layers */
--surface: #1A1F2E;
--surface-alt: #222838;
--surface-raised: #2A3142;
```

**Step 4: Commit**

```bash
cd /home/pedrad/javier_project_folder/onmisense/Mclarenpredictivemaintenanceplatform
git add src/styles/theme.css
git commit -m "fix(tokens): unify palette to canonical values, neutral borders"
```

---

### Task 3: Repo A Components — Replace Hardcoded Border Classes

**Files (30 component files in Repo A):**
All `.tsx` files in `/home/pedrad/javier_project_folder/f1/frontend/src/app/components/`

**Step 1: Replace resting orange borders with token reference**

Find-and-replace across all component files:

| Find | Replace | Context |
|------|---------|---------|
| `border-[rgba(255,128,0,0.12)]` | `border-border` | Resting card/container borders |
| `border-[rgba(255,128,0,0.08)]` | `border-border` | Lighter resting borders |
| `border-[rgba(255,128,0,0.06)]` | `border-border` | Divider borders |
| `border-[rgba(255,128,0,0.1)]` | `border-border` | Variant resting borders |

Do NOT replace:
- `border-[rgba(255,128,0,0.2)]` — this is hover/tooltip (signal state, use `border-border-strong` if desired)
- `border-[rgba(255,128,0,0.25)]` — signal accent borders
- `border-[rgba(255,128,0,0.3)]` or higher — active/signal states
- `border-[#FF8000]` — explicit active state borders
- `hover:border-*` with orange — these are interactive signal states

**Step 2: Replace hardcoded card backgrounds with token**

| Find | Replace |
|------|---------|
| `bg-[#1A1F2E]` | `bg-card` |
| `bg-[#1a1f2e]` | `bg-card` |

**Step 3: Replace hardcoded page backgrounds with token**

| Find | Replace |
|------|---------|
| `bg-[#0D1117]` | `bg-background` |
| `bg-[#0d1117]` | `bg-background` |

**Step 4: Replace hardcoded secondary backgrounds with token**

| Find | Replace |
|------|---------|
| `bg-[#222838]` | `bg-secondary` |

**Step 5: Commit**

```bash
cd /home/pedrad/javier_project_folder/f1
git add frontend/src/app/components/
git commit -m "refactor(ui): replace hardcoded colors with design tokens in Repo A"
```

---

### Task 4: Repo A Components — Neutral Chart Grid Strokes

**Files (chart-heavy components):**
- `DriverBiometrics.tsx` (~22 instances)
- `CarTelemetry.tsx` (~20 instances)
- `RaceStrategy.tsx` (~8 instances)
- `CircuitIntel.tsx`
- `McLarenAnalytics.tsx`
- `LiveDashboard.tsx`
- `ForecastChart.tsx`
- `KexBriefingCard.tsx`
- `BacktestView.tsx`
- `DriverIntel.tsx`

**Step 1: Replace orange chart grid strokes with neutral**

In all `<CartesianGrid>` components, replace:
```tsx
/* BEFORE */
stroke="rgba(255,128,0,0.12)"
stroke="rgba(255,128,0,0.08)"
stroke="rgba(255,128,0,0.06)"
stroke="rgba(255, 128, 0, 0.12)"
stroke="rgba(255, 128, 0, 0.08)"
stroke="rgba(255, 128, 0, 0.06)"

/* AFTER — all become: */
stroke="rgba(255,255,255,0.05)"
```

**Step 2: Replace orange tooltip border strokes with neutral**

In custom tooltip `contentStyle` objects, replace:
```tsx
/* BEFORE */
border: '1px solid rgba(255,128,0,0.2)'
border: "1px solid rgba(255,128,0,0.2)"

/* AFTER */
border: '1px solid rgba(255,255,255,0.1)'
```

**Step 3: Commit**

```bash
cd /home/pedrad/javier_project_folder/f1
git add frontend/src/app/components/
git commit -m "refactor(charts): neutral grid lines and tooltip borders"
```

---

### Task 5: Repo B Components — Replace Hardcoded Colors

**Files (10 component files in Repo B):**
All `.tsx` files in `/home/pedrad/javier_project_folder/onmisense/Mclarenpredictivemaintenanceplatform/src/app/`

**Step 1: Replace resting orange borders with token reference**

| Find | Replace |
|------|---------|
| `border-[rgba(255,128,0,0.12)]` | `border-border` |
| `border-[rgba(255,128,0,0.08)]` | `border-border` |
| `border-[rgba(255,128,0,0.06)]` | `border-border` |
| `border-[rgba(255,128,0,0.1)]` | `border-border` |

Same exclusion rules as Task 3.

**Step 2: Replace hardcoded card backgrounds**

| Find | Replace |
|------|---------|
| `bg-[#12121e]` | `bg-card` |
| `bg-[#0a0a12]` | `bg-background` |
| `bg-[#1a1a2e]` | `bg-secondary` |
| `bg-[#0d0d18]` | `bg-background` |

**Step 3: Replace chart grid strokes**

Same as Task 4 — all `CartesianGrid` and tooltip borders to neutral.

**Step 4: Commit**

```bash
cd /home/pedrad/javier_project_folder/onmisense/Mclarenpredictivemaintenanceplatform
git add src/
git commit -m "refactor(ui): replace hardcoded colors with design tokens, unify palette"
```

---

### Task 6: Verify Both Repos Build & Render

**Step 1: Build Repo A**

```bash
cd /home/pedrad/javier_project_folder/f1/frontend
npm run build
```

Expected: Clean build, no errors. Warnings about unused imports are acceptable.

**Step 2: Build Repo B**

```bash
cd /home/pedrad/javier_project_folder/onmisense/Mclarenpredictivemaintenanceplatform
npm run build
```

Expected: Clean build, no errors.

**Step 3: Visual spot-check**

Start dev servers and verify:
- Card borders are now subtle neutral white (not orange-tinted)
- Chart grid lines are neutral (not orange)
- Active nav items still show `#FF8000` orange
- Status badges still show correct green/amber/red
- Hover states on buttons still show orange accent
- Overall dark theme looks cohesive

**Step 4: Final commit if any fixups needed**

```bash
git commit -m "fix: post-token-migration visual fixups"
```

---

## Summary of Changes

| What | Before | After | Impact |
|------|--------|-------|--------|
| Resting borders | Orange `rgba(255,128,0,0.12)` | Neutral `rgba(255,255,255,0.06)` | Orange reserved for signals |
| Chart grids | Orange-tinted | Neutral white at 5% | Charts feel analytical |
| Card backgrounds | Hardcoded hex per repo | `bg-card` token | Single source of truth |
| Page backgrounds | Hardcoded hex per repo | `bg-background` token | Unified across repos |
| Repo B palette | Drifted values | Matches Repo A exactly | One design system |
| Total files touched | ~42 | — | Token-only, no logic changes |

## Risk Assessment

**Risk: Near zero.** All changes are CSS color values. No component props, state, or DOM structure changes. If any color looks wrong, it can be reverted by changing the CSS variable value in theme.css — one line fix.
