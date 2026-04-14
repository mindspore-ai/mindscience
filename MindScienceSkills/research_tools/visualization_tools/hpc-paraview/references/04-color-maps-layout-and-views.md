# ParaView Color Maps Layout And Views

## Purpose

Use this reference when tuning representations, color maps, layouts, or screenshot output.

## View logic

High-value checks:

- active view is the intended one
- camera is meaningful for the data
- representation type matches the data and task

## Coloring rules

Before trusting a colored result:

- confirm which array is active
- confirm whether point or cell data is being used
- confirm the color range is meaningful

## Layout and screenshot rules

Use layout-aware saves when multiple views matter. Use single-view saves when only one view is authoritative.

Portable scripting pattern:

- get the intended view or layout explicitly
- save the screenshot with explicit resolution when reproducibility matters

## Failure patterns

| Symptom | Likely cause | First repair |
| --- | --- | --- |
| screenshot is not the view you expected | wrong active view or layout object | save from an explicit view or layout handle |
| colors look misleading | wrong array or bad range | restate active array and color-range intent |
| layout export differs between runs | implicit UI state | make view size and target object explicit |
