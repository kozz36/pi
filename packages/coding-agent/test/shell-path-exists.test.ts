import type * as ChildProcess from "node:child_process";
import { chmodSync, mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import type * as Fs from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { getPowerShellConfig, getShellConfig } from "../src/utils/shell.ts";

const { accessSyncMock, statSyncMock, spawnSyncMock } = vi.hoisted(() => ({
	accessSyncMock: vi.fn(),
	statSyncMock: vi.fn(),
	spawnSyncMock: vi.fn(),
}));

vi.mock("node:fs", async (importOriginal) => ({
	...(await importOriginal<typeof Fs>()),
	accessSync: accessSyncMock,
	statSync: statSyncMock,
}));

vi.mock("child_process", async (importOriginal) => ({
	...(await importOriginal<typeof ChildProcess>()),
	spawnSync: spawnSyncMock,
}));

const platformDescriptor = Object.getOwnPropertyDescriptor(process, "platform")!;
const aliasesDir = "D:\\Users\\test\\AppData\\Local\\Microsoft\\WindowsApps";

function stubPlatform(platform: string) {
	Object.defineProperty(process, "platform", { configurable: true, value: platform });
}

function available(...paths: string[]) {
	accessSyncMock.mockImplementation((path: string) => {
		if (!paths.includes(path)) throw Object.assign(new Error("not found"), { code: "ENOENT" });
	});
}

beforeEach(() => {
	vi.stubEnv("ProgramFiles", undefined);
	vi.stubEnv("ProgramFiles(x86)", undefined);
	vi.stubEnv("SystemRoot", undefined);
	vi.stubEnv("PATH", "");
	accessSyncMock.mockReset();
	statSyncMock.mockReset().mockReturnValue({ isFile: () => true });
	spawnSyncMock.mockReset().mockReturnValue({ status: 1, stdout: "" });
});

afterEach(() => {
	Object.defineProperty(process, "platform", platformDescriptor);
	vi.unstubAllEnvs();
});

// nodejs/node#36790: runnable Store aliases pass access(F_OK) but fail the extra
// stat check existsSync performs on Windows only.
describe("Windows Store shell aliases", () => {
	beforeEach(() => {
		stubPlatform("win32");
		vi.stubEnv("PATH", aliasesDir);
	});

	it.each(["bash.exe", "pwsh.exe"])("accepts %s found on PATH without discovery processes or stat", (executable) => {
		const alias = `${aliasesDir}\\${executable}`;
		available(alias);

		const config = executable === "pwsh.exe" ? getPowerShellConfig() : getShellConfig();

		expect(config.shell).toBe(alias);
		expect(accessSyncMock).toHaveBeenCalledWith(alias, 0);
		expect(statSyncMock).not.toHaveBeenCalled();
		expect(spawnSyncMock).not.toHaveBeenCalled();
	});

	it("accepts an explicitly configured shell alias", () => {
		const alias = `${aliasesDir}\\bash.exe`;
		available(alias);
		expect(getShellConfig(alias).shell).toBe(alias);
		expect(statSyncMock).not.toHaveBeenCalled();
		expect(spawnSyncMock).not.toHaveBeenCalled();
	});

	it("still rejects PATH results that fail the existence check", () => {
		const windowsPowerShell = "D:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe";
		vi.stubEnv("PATH", `${aliasesDir};D:\\Windows\\System32\\WindowsPowerShell\\v1.0`);
		available(windowsPowerShell);

		expect(getPowerShellConfig().shell).toBe(windowsPowerShell);
		expect(spawnSyncMock).not.toHaveBeenCalled();
	});

	it("preserves quoted PATH entries and first available entry", () => {
		vi.stubEnv("PATH", '"D:\\First Tools";E:\\Second');
		available("D:\\First Tools\\bash.exe", "E:\\Second\\bash.exe");
		expect(getShellConfig().shell).toBe("D:\\First Tools\\bash.exe");
		expect(spawnSyncMock).not.toHaveBeenCalled();
	});

	it("searches the current directory before Windows PATH, as where did", () => {
		const cwd = vi.spyOn(process, "cwd").mockReturnValue("E:\\Work");
		try {
			available("E:\\Work\\bash.exe", `${aliasesDir}\\bash.exe`);
			expect(getShellConfig().shell).toBe("E:\\Work\\bash.exe");
			expect(spawnSyncMock).not.toHaveBeenCalled();
		} finally {
			cwd.mockRestore();
		}
	});

	it("prefers known Git Bash locations to PATH", () => {
		vi.stubEnv("ProgramFiles", "D:\\Apps");
		vi.stubEnv("ProgramFiles(x86)", "E:\\Apps");
		available("E:\\Apps\\Git\\bin\\bash.exe", `${aliasesDir}\\bash.exe`);
		expect(getShellConfig().shell).toBe("E:\\Apps\\Git\\bin\\bash.exe");
		expect(spawnSyncMock).not.toHaveBeenCalled();
	});

	it("reports unavailable Bash without invoking a helper", () => {
		available();
		expect(() => getShellConfig()).toThrow("No bash shell found");
		expect(spawnSyncMock).not.toHaveBeenCalled();
	});
});

describe("Unix shell existence checks", () => {
	beforeEach(() => {
		stubPlatform("linux");
	});

	it("accepts /bin/bash when access(F_OK) succeeds", () => {
		expect(getShellConfig().shell).toBe("/bin/bash");
		expect(spawnSyncMock).not.toHaveBeenCalled();
	});

	it("falls back to executable bash on PATH when /bin/bash fails access(F_OK)", () => {
		vi.stubEnv("PATH", "/not-executable:/usr/local/bin:/later");
		available("/usr/local/bin/bash", "/later/bash");

		expect(getShellConfig().shell).toBe("/usr/local/bin/bash");
		expect(accessSyncMock).toHaveBeenCalledWith("/usr/local/bin/bash", 1);
		expect(spawnSyncMock).not.toHaveBeenCalled();
	});

	it.each(["", "."])("keeps a current-directory PATH match explicit for PATH=%j", (path) => {
		vi.stubEnv("PATH", path);
		available("./bash");
		expect(getShellConfig().shell).toBe("./bash");
		expect(spawnSyncMock).not.toHaveBeenCalled();
	});

	it("falls back to sh when PATH has no executable bash", () => {
		vi.stubEnv("PATH", "/missing");
		available();
		expect(getShellConfig()).toEqual({ shell: "sh", args: ["-c"] });
		expect(spawnSyncMock).not.toHaveBeenCalled();
	});

	it("rejects an explicitly configured shell path that fails access(F_OK)", () => {
		available();
		expect(() => getShellConfig("/missing/bash")).toThrow("Custom shell path not found: /missing/bash");
		expect(spawnSyncMock).not.toHaveBeenCalled();
	});

	it.skipIf(process.platform === "win32")("checks executable POSIX files, directories and symlinks", async () => {
		const fs = await vi.importActual<typeof Fs>("node:fs");
		const root = mkdtempSync(join(tmpdir(), "shell-discovery-"));
		const paths = ["nonexec", "directory", "executable"].map((name) => join(root, name));
		for (const path of paths) {
			mkdirSync(path, { mode: 0o755 });
			chmodSync(path, 0o755);
		}
		writeFileSync(join(paths[0], "bash"), "fixture", { mode: 0o644 });
		chmodSync(join(paths[0], "bash"), 0o644);
		mkdirSync(join(paths[1], "bash"), { mode: 0o755 });
		chmodSync(join(paths[1], "bash"), 0o755);
		const target = join(root, "target");
		writeFileSync(target, "fixture", { mode: 0o644 });
		chmodSync(target, 0o755);
		fs.symlinkSync(target, join(paths[2], "bash"));
		vi.stubEnv("PATH", paths.join(":"));
		accessSyncMock.mockImplementation((path, mode) => {
			if (path === "/bin/bash") throw new Error("controlled fallback");
			return fs.accessSync(path, mode);
		});
		statSyncMock.mockImplementation(fs.statSync);

		expect(getShellConfig().shell).toBe(join(paths[2], "bash"));
		chmodSync(target, 0o644);
		expect(getShellConfig()).toEqual({ shell: "sh", args: ["-c"] });
		expect(fs.readFileSync(target, "utf8")).toBe("fixture");
		expect(spawnSyncMock).not.toHaveBeenCalled();
	});
});
