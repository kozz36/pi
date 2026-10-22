import type * as ChildProcess from "node:child_process";
import type * as Fs from "node:fs";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { getPowerShellConfig, POWERSHELL_ARGS } from "../src/utils/shell.ts";

const { accessSyncMock, spawnSyncMock } = vi.hoisted(() => ({
	accessSyncMock: vi.fn(),
	spawnSyncMock: vi.fn(),
}));

vi.mock("node:fs", async (importOriginal) => ({
	...(await importOriginal<typeof Fs>()),
	accessSync: accessSyncMock,
}));

vi.mock("child_process", async (importOriginal) => ({
	...(await importOriginal<typeof ChildProcess>()),
	spawnSync: spawnSyncMock,
}));

const platformDescriptor = Object.getOwnPropertyDescriptor(process, "platform")!;
const pwsh7 = "D:\\Applications\\PowerShell\\7\\pwsh.exe";
const windowsPowerShell = "E:\\WinNT\\System32\\WindowsPowerShell\\v1.0\\powershell.exe";

beforeEach(() => {
	Object.defineProperty(process, "platform", { configurable: true, value: "win32" });
	vi.stubEnv("ProgramFiles", "D:\\Applications");
	vi.stubEnv("SystemRoot", "E:\\WinNT");
	vi.stubEnv("PATH", "E:\\Tools");
	accessSyncMock.mockReset().mockImplementation(() => {
		throw Object.assign(new Error("not found"), { code: "ENOENT" });
	});
	spawnSyncMock.mockReset().mockReturnValue({ status: 1, stdout: "" });
});

afterEach(() => {
	Object.defineProperty(process, "platform", platformDescriptor);
	vi.unstubAllEnvs();
	expect(spawnSyncMock).not.toHaveBeenCalled();
});

// #9490: discover PowerShell outside PATH without assuming a drive or Windows directory name.
describe("getPowerShellConfig", () => {
	it("prefers PowerShell 7 on PATH over known installation locations", () => {
		const shell = "E:\\Tools\\pwsh.exe";
		accessSyncMock.mockImplementation((path: string) => {
			if (path !== shell) throw new Error("not found");
		});

		expect(getPowerShellConfig()).toEqual({ shell, args: [...POWERSHELL_ARGS] });
	});

	it("finds PowerShell 7 under ProgramFiles before Windows PowerShell on PATH", () => {
		vi.stubEnv("PATH", "E:\\WinNT\\System32\\WindowsPowerShell\\v1.0");
		accessSyncMock.mockImplementation((path: string) => {
			if (![pwsh7, windowsPowerShell].includes(path as string)) {
				throw Object.assign(new Error("not found"), { code: "ENOENT" });
			}
		});

		expect(getPowerShellConfig()).toEqual({ shell: pwsh7, args: [...POWERSHELL_ARGS] });
	});

	it("prefers Windows PowerShell on PATH over its system installation", () => {
		const shell = "E:\\Tools\\powershell.exe";
		accessSyncMock.mockImplementation((path: string) => {
			if (![shell, windowsPowerShell].includes(path as string)) {
				throw Object.assign(new Error("not found"), { code: "ENOENT" });
			}
		});

		expect(getPowerShellConfig()).toEqual({ shell, args: [...POWERSHELL_ARGS] });
	});

	it("finds Windows PowerShell under SystemRoot when PATH lookup is unavailable", () => {
		vi.stubEnv("PATH", undefined);
		accessSyncMock.mockImplementation((path: string) => {
			if (path !== windowsPowerShell) {
				throw Object.assign(new Error("not found"), { code: "ENOENT" });
			}
		});

		expect(getPowerShellConfig()).toEqual({ shell: windowsPowerShell, args: [...POWERSHELL_ARGS] });
	});

	it("reports an error when no PowerShell installation exists", () => {
		expect(() => getPowerShellConfig()).toThrow("No PowerShell executable found");
	});
});
