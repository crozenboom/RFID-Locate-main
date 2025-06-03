# Impinj R420 + Sllurp Reader Settings Reference

This document summarizes key reader settings available through the **Sllurp** LLRP client for the **Impinj Speedway R420** RFID reader. Use this as a guide to tune your reader for different use cases like inventory snapshots, real-time tracking, or portal monitoring.

---


## Inventory Session (`session`)

Defines how long a tag remains silent after responding to a query.

| Value | Name | Persistence | Description |
|-------|------|-------------|-------------|
| 0     | S0   | Very Short  | Tag responds again almost immediately. Best for high-speed environments (e.g., conveyors). |
| 1     | S1   | Medium      | Default session; good balance for general use. |
| 2     | S2   | Long        | Tag stays quiet longer after being read. Good for reducing duplicates. |
| 3     | S3   | Longest     | Longest tag quiet time; ideal for inventory counts where re-reads are undesirable. |

---

## Search Mode (`impinj_search_mode`)

Controls how the reader searches and interacts with tags in different states.

| Value | Name          | Description |
|-------|---------------|-------------|
| 1     | `SingleTarget`| Fast, reads tags in only one session state (A or B). May miss some tags. |
| 2     | `DualTarget`  | Alternates between A and B — improves tag discovery but adds overhead. |
| 3     | `TagFocus`    | Optimized for re-reads of the same tag. Helps prevent false positives. |

---

## Report Frequency (`report_every_n_tags`)

Controls how often tag reports are emitted.

| Value | Description |
|-------|-------------|
| 1     | Report every tag read (default) |
| n > 1 | Report every nth tag seen. Helps reduce network traffic and processing. |

---

## Mode Identifier (`mode_identifier`)

Defines the **RF operating mode** used by the reader. This affects read speed, interference tolerance, and how the reader behaves in dense environments.

---

### Standard Modes (0–5)

| Value | Name                  | Notes                                                                 |
|-------|-----------------------|-----------------------------------------------------------------------|
| 0     | Max Throughput        | Fastest data rate, but least interference tolerant                    |
| 1     | Hybrid                | Balanced speed and noise handling                                     |
| 2     | Dense Reader M4       | Standard interference protection                                      |
| 3     | Dense Reader M8       | Most interference-tolerant standard mode                              |
| 4     | Max Miller            | High read rate with moderate interference tolerance                   |
| 5     | Dense Reader M4 Two   | Faster forward link than Mode 2; good for multi-reader environments   |

---

### Impinj AutoSet Modes (1000+)

| Value | Name                             | Notes                                                                 |
|-------|----------------------------------|-----------------------------------------------------------------------|
| 1000  | AutoSet Dense Reader             | Dynamically selects Mode 1–5 based on RF environment (patented)      |
| 1002  | AutoSet Dense Reader Deep Scan   | Combines fast and dense modes; maximizes unique tag reads            |
| 1003  | AutoSet Static Fast              | Fast modes only; good read rate with some interference tolerance     |
| 1004  | AutoSet Static Dense Reader      | Dense modes only; ideal for crowded RF environments                  |
| 1005  | Impinj Internal                  | ⚠️ Reserved — **Do Not Use**                                          |

---

**Tips:**
- Use **0–5** for full manual control of performance and noise tolerance.
- Use **1000–1004** when you want the reader to adapt to the RF environment automatically.