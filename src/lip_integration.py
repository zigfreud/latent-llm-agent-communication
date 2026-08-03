import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.core.hidden_states import select_hidden_vectors
from src.core.prompt_protocol import format_prompt, tokenizer_add_special_tokens
from src.lip_protocol import LIPPacket

DEVICE = "cpu"
PROMPT_PROTOCOL = {
    "version": "lip-prompt-v1",
    "mode": "chat_template",
    "add_generation_prompt": True,
    "system_prompt": None,
}
ADAPTER_PATH = os.path.join("experiments", "experiments_log", r"20251123_1819_Gen4_Code_Specialist", "final_adapter_gen5.pth")

print(f"🔧 [LIP Integration] Starting Universal Protocol Demo...")


class LIPSender:
    def __init__(self, adapter_path):
        print("\n📱 Initializing Client (TinyLlama)...")
        self.model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, device_map=DEVICE, torch_dtype=torch.float32, low_cpu_mem_usage=True
        )
        self.model.eval()

        self.adapter = torch.nn.Sequential(
            torch.nn.Linear(2048, 512),
            torch.nn.LayerNorm(512),
            torch.nn.GELU(),
            torch.nn.Linear(512, 4096),
            torch.nn.LayerNorm(4096)
        ).to(DEVICE)

        try:
            state = torch.load(adapter_path, map_location=DEVICE, weights_only=True)
            # Mapping HeteroAdapter (named layers) to Sequential (indexed layers) if necessary
            # Ideally, define the class structure identically.
            try:
                self.adapter.load_state_dict(state)
            except RuntimeError:
                # Quick fix for structure mismatch between class and sequential
                from collections import OrderedDict
                new_state = OrderedDict()
                new_state['0.weight'] = state['encoder.0.weight']
                new_state['0.bias'] = state['encoder.0.bias']
                new_state['1.weight'] = state['encoder.1.weight']
                new_state['1.bias'] = state['encoder.1.bias']
                new_state['3.weight'] = state['decoder.0.weight']
                new_state['3.bias'] = state['decoder.0.bias']
                new_state['4.weight'] = state['decoder.1.weight']
                new_state['4.bias'] = state['decoder.1.bias']
                self.adapter.load_state_dict(new_state)

            print("✅ Adapter Loaded on Client.")
        except Exception as e:
            raise RuntimeError(f"could not safely load demo adapter: {e}") from e


    def create_packet(self, prompt):
        print(f"   Thinking: '{prompt}'...")
        with torch.no_grad():
            formatted = format_prompt(prompt, self.tokenizer, PROMPT_PROTOCOL)
            inp = self.tokenizer(
                formatted,
                return_tensors="pt",
                add_special_tokens=tokenizer_add_special_tokens(PROMPT_PROTOCOL),
            ).to(DEVICE)
            outputs = self.model(**inp, output_hidden_states=True)
            vec_src = select_hidden_vectors(
                outputs.hidden_states[-1],
                inp.get("attention_mask"),
                token_position="last_non_padding",
            ).to(torch.float32)
            vec_trans = self.adapter(vec_src)
            packet = LIPPacket(vec_trans, source_model="TinyLlama+AdapterV1", intent="code")
            return packet.to_json()


class LIPReceiver:
    def __init__(self):
        print("\n☁️  Initializing Server (Llama-3)...")
        self.model_id = "NousResearch/Meta-Llama-3-8B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        
        print(f"   Trying to load model on {DEVICE}...")
        try:
            # Tentativa 1: Otimizada (bfloat16)
            print("   [1/2] Attempting bfloat16 load (Low RAM)...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, 
                device_map=DEVICE, 
                torch_dtype=torch.bfloat16, 
                low_cpu_mem_usage=True
            )
            print("   ✅ Loaded in bfloat16.")
        except Exception as e:
            print(f"   ⚠️ bfloat16 failed: {e}")
            print("   [2/2] Attempting float32 load (High RAM usage!)...")
            # Tentativa 2: Fallback explícito
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id, 
                    device_map=DEVICE, 
                    torch_dtype=torch.float32
                )
                print("   ✅ Loaded in float32.")
            except Exception as e2:
                print(f"   ❌ FATAL: Could not load model. Error: {e2}")
                sys.exit(1) # Mata o processo se falhar

        self.model.eval()

        print("   Calculating Energy Signature...")
        with torch.no_grad():
            self.ref_energy = self.model.get_input_embeddings().weight.norm(p=2, dim=-1).mean().item()
            print(f"⚡ Server Reference Energy: {self.ref_energy:.4f}")

    def process_packet(self, json_packet):

        print(f"📩 Packet Received. Size: {len(json_packet)} bytes.")
        packet = LIPPacket.from_json(json_packet)

        print(f"Origin: {packet.header.source_model} | ID: {packet.header.packet_id[:8]}...")

        vector = packet.unpack_vector(device=DEVICE)
        calibrated_vec = packet.calibrate_energy(self.ref_energy, vector)
        self._inject_and_generate(calibrated_vec)

    def _inject_and_generate(self, vector):
        bos_id = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id else self.tokenizer.eos_token_id
        bos_tensor = torch.tensor([[bos_id]], device=DEVICE)

        # Context Priming
        priming_text = "Here is the Python code snippet:\n"
        priming_ids = self.tokenizer(priming_text, return_tensors="pt", add_special_tokens=False).input_ids.to(DEVICE)

        with torch.no_grad():
            bos_embed = self.model.get_input_embeddings()(bos_tensor).to(self.model.dtype)
            priming_embed = self.model.get_input_embeddings()(priming_ids).to(self.model.dtype)
            vec_final = vector.unsqueeze(1).to(self.model.dtype)

            # Sandwich: [BOS] + [PRIMING] + [VECTOR]
            input_embeds = torch.cat([bos_embed, priming_embed, vec_final], dim=1)
            att_mask = torch.ones(input_embeds.shape[:2], dtype=torch.long, device=DEVICE)

            print(f"🤖 Context Activated: '{priming_text.strip()}'")
            print("🤖 Generating response...")

            gen_ids = self.model.generate(
                inputs_embeds=input_embeds,
                attention_mask=att_mask,
                max_new_tokens=60,
                do_sample=True,
                temperature=0.1,
                top_p=0.9,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.eos_token_id
            )

            output = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
            print(f"\n🟢 LIP RESPONSE: {output}\n")


if __name__ == "__main__":
    print(f"🔎 Checking Adapter at: {ADAPTER_PATH}")
    if not os.path.exists(ADAPTER_PATH):
        print(f"❌ Adapter NOT FOUND. Check the path/filename.")
        print(f"   Current Dir: {os.getcwd()}")
        sys.exit(1)
    
    print("✅ Adapter found. Starting sequence.")
    
    try:
        sender = LIPSender(ADAPTER_PATH)
    except Exception as e:
        print(f"❌ Sender Init Failed: {e}")
        sys.exit(1)
        
    try:
        receiver = LIPReceiver()
    except Exception as e:
        print(f"❌ Receiver Init Failed: {e}")
        sys.exit(1)

    prompt = "write an python hello world function"

    print("-" * 60)
    print(f"📝 Input Prompt: '{prompt}'")
    
    try:
        json_payload = sender.create_packet(prompt)
        print(f"\n🌐 [NETWORK] Transmitting JSON: {len(json_payload)} bytes...")
        receiver.process_packet(json_payload)
    except Exception as e:
        print(f"❌ Runtime Error during transmission: {e}")
        
    print("-" * 60)
