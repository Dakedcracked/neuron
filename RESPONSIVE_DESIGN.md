# Responsive Design Implementation — Neuron AI v2.0

## Overview
Professional mobile-first responsive design with proper breakpoints, z-index stacking, and layout flow optimization for all device sizes.

## Breakpoints & Strategy

### 1. **Desktop (> 1024px)**
- Full sidebar (280px fixed)
- Main content has 24px padding on desktop
- All grids visible at full width
- Multi-column layouts active

### 2. **Tablet (≤ 1024px)**
- Sidebar transforms to slide-out (220px width, 80vw max)
- Mobile topbar appears (56px fixed)
- Main content: 72px top padding (56px bar + 16px gap)
- 2-column stats/grids
- Controls wrapped but still visible

### 3. **Mobile (≤ 768px)**
- Sidebar full-width offscreen
- Mobile topbar: 56px (permanent)
- Main content: 68px top padding
- 2-column stats, single-column everything else
- Tables horizontally scrollable
- All forms full-width

### 4. **Small Phone (≤ 520px)**
- Sidebar: 100% width, full-height offscreen overlay
- Mobile topbar: 52px (optimized for thumbs)
- Main content: 64px top padding
- All grids: 1 column
- Typography scaled down 20-30%
- Touch targets: minimum 34-36px
- Zero horizontal overflow

## Key CSS Fixes Applied

### Layout Stacking (`z-index` hierarchy)
```
Sidebar:                z-index: 40 (highest, transforms out of view)
Mobile topbar:          z-index: 38
Sidebar backdrop:       z-index: 30 (clickable overlay)
Main content:           z-index: auto (default, below sidebar)
```

### Margin & Padding Calculations
- **Desktop**: `main-content { margin-left: 280px; padding: 24px 28px; }`
- **Tablet (1024px)**: `main-content { margin-left: 0; padding: 72px 20px 20px; }`
- **Mobile (768px)**: `main-content { margin-left: 0; padding: 68px 16px 16px; }`
- **Small (520px)**: `main-content { padding: 64px 12px 12px; }`

All use `box-sizing: border-box` to prevent overflow.

### Grid Cascading
```css
/* Desktop */
.stat-grid { grid-template-columns: repeat(4, 1fr); }

/* Tablet */
@media (max-width: 1024px) {
  .stat-grid { grid-template-columns: repeat(2, 1fr); }
}

/* Mobile */
@media (max-width: 768px) {
  .stat-grid { grid-template-columns: repeat(2, 1fr); }
}

/* Small phone */
@media (max-width: 520px) {
  .stat-grid { grid-template-columns: 1fr; }
}
```

### Table Overflow Handling
- `overflow-x: auto` on `.table-scroll`
- `-webkit-overflow-scrolling: touch` for iOS momentum scrolling
- Font size reduced 20-30% on mobile to fit more columns
- Min-width on tables to prevent content collapse

### Touch Optimization
- Buttons: minimum 36px (44px ideal per WCAG, 36px minimum on constrained screens)
- No hover effects on touch devices (handled via CSS)
- Tap targets have adequate spacing (8px gap minimum)
- Avoid double-tap zoom by managing viewport

## JavaScript Sidebar Toggle

```javascript
function toggleSidebar() {
    const sidebar = document.getElementById('sidebar');
    const backdrop = document.getElementById('sidebar-backdrop');
    sidebar.classList.toggle('open');
    backdrop.classList.toggle('visible');
    document.body.style.overflow = sidebar.classList.contains('open') ? 'hidden' : 'auto';
}

function closeSidebar() {
    const sidebar = document.getElementById('sidebar');
    const backdrop = document.getElementById('sidebar-backdrop');
    sidebar.classList.remove('open');
    backdrop.classList.remove('visible');
    document.body.style.overflow = 'auto';
}

// Close sidebar when backdrop clicked
document.getElementById('sidebar-backdrop')?.addEventListener('click', closeSidebar);

// Close sidebar when nav item clicked (in navigateTo function)
if (window.innerWidth <= 1024) closeSidebar();
```

## Critical Fixes Applied in v2.0

1. **Fixed layout collapse**: `width: 100%` + `box-sizing: border-box` on main-content
2. **Fixed topbar overlap**: Proper `padding-top` calculations for each breakpoint
3. **Fixed sidebar z-index**: Proper stacking context and transform behavior
4. **Fixed table overflow**: Horizontal scroll with touch support
5. **Fixed touch targets**: Minimum 36px for mobile buttons
6. **Fixed typography scale**: No orphaned text, proper cascading font sizes
7. **Fixed grid stacking**: Proper `!important` flags where cascade conflicts exist
8. **Fixed backdrop interaction**: Click-to-close and scroll prevention

## Testing Checklist

- [ ] Desktop (1440px): Sidebar visible, full layout
- [ ] Tablet (1024px): Sidebar togglable, 2-column grids
- [ ] Tablet landscape (850px): Single topbar, grids adjust
- [ ] Mobile (768px): 2-column stats, single-column content
- [ ] Mobile (600px): All single-column, readable typography
- [ ] Small phone (520px): Touch-friendly buttons, no horizontal scroll
- [ ] Touch interactions: Sidebar toggle works, backdrop clickable
- [ ] Tables: Horizontal scroll on mobile, readable font
- [ ] Forms: Full-width inputs, proper spacing
- [ ] Dropzone: Proper sizing at all breakpoints
- [ ] Viewer: Image scaling, bbox visible, no overflow

## Future Enhancements

- Add dark mode responsive adjustments
- Test on actual devices (iOS Safari, Chrome Mobile)
- Add landscape-only media queries if needed
- Consider adding print styles
- Add reduced-motion support for animations
